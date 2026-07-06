"""Day 2 - bidirectional-LSTM encoder and Bahdanau (additive) attention.

Day 1 gave us padded batches of shape (batch, time) plus the true source
lengths. Today we turn a source batch into a sequence of context-aware hidden
states and build the attention module the decoder will query each step.

Two pieces:

  1. Encoder - an embedding followed by a *bidirectional* LSTM. Running the LSTM
     both directions means every source position's representation sees both its
     left and right context, which matters a lot for word order differences
     between languages. We pack the padded batch by its real lengths so the LSTM
     never wastes computation (or state) on <pad> steps.

  2. Bahdanau attention - an additive score between the current decoder state and
     every encoder output, turned into a softmax weighting and then a context
     vector. Crucially we mask padded source positions to -inf *before* the
     softmax so they get zero weight; skip that and the alignments smear onto
     padding and translation quality collapses.

Day 3 wires an attention decoder on top of these with teacher forcing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from day1_data import build_dataloaders


class Encoder(nn.Module):
    """Embed the source tokens and run a bidirectional LSTM over them.

    The two directions are concatenated per timestep, so each encoder output has
    width ``2 * hidden_size``. The final forward/backward hidden and cell states
    are combined through a small linear so the (unidirectional) decoder can be
    initialised from them.
    """

    def __init__(self, vocab_size, emb_dim, hidden_size, pad_id, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(
            emb_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # fold the concatenated forward+backward states down to decoder width
        self.bridge_h = nn.Linear(2 * hidden_size, hidden_size)
        self.bridge_c = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, src, src_lens):
        # src: (batch, src_time) of token ids; src_lens: (batch,) true lengths
        embedded = self.embedding(src)  # (batch, src_time, emb_dim)

        # pack so the LSTM stops at each sequence's real length, not the padding.
        # enforce_sorted=False lets us hand batches in arbitrary length order.
        packed = pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h, c) = self.rnn(packed)
        # (batch, src_time, 2 * hidden_size)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True)

        # h/c are (num_layers * 2, batch, hidden). Grab the last layer's forward
        # and backward states, concatenate, and bridge to the decoder's size.
        h_dec = self._bridge(h, self.bridge_h)
        c_dec = self._bridge(c, self.bridge_c)
        return outputs, (h_dec, c_dec)

    def _bridge(self, state, layer):
        # state: (num_layers * 2, batch, hidden). Reshape to expose the direction
        # axis, take the last layer, concat the two directions, then project.
        state = state.view(self.num_layers, 2, state.size(1), self.hidden_size)
        last = state[-1]  # (2, batch, hidden) -> forward, backward
        combined = torch.cat([last[0], last[1]], dim=1)  # (batch, 2*hidden)
        return torch.tanh(layer(combined)).unsqueeze(0)  # (1, batch, hidden)


class BahdanauAttention(nn.Module):
    """Additive attention scoring a decoder state against all encoder outputs.

    score(dec, enc_t) = v^T tanh(W_dec * dec + W_enc * enc_t)

    The encoder-output projection does not depend on the decoder state, so it can
    be precomputed once per source sentence and reused across all decode steps.
    """

    def __init__(self, dec_hidden, enc_hidden):
        super().__init__()
        # encoder outputs are bidirectional, hence 2 * enc_hidden wide
        self.W_enc = nn.Linear(2 * enc_hidden, dec_hidden, bias=False)
        self.W_dec = nn.Linear(dec_hidden, dec_hidden, bias=False)
        self.v = nn.Linear(dec_hidden, 1, bias=False)

    def forward(self, dec_state, enc_outputs, src_mask):
        # dec_state: (batch, dec_hidden); enc_outputs: (batch, src_time, 2*enc_hidden)
        # src_mask: (batch, src_time) with True on real tokens, False on padding
        proj_enc = self.W_enc(enc_outputs)              # (batch, src_time, dec_hidden)
        proj_dec = self.W_dec(dec_state).unsqueeze(1)   # (batch, 1, dec_hidden)
        scores = self.v(torch.tanh(proj_enc + proj_dec)).squeeze(-1)  # (batch, src_time)

        # kill padded positions before the softmax so they get zero weight
        scores = scores.masked_fill(~src_mask, float("-inf"))
        weights = F.softmax(scores, dim=1)              # (batch, src_time)

        # weighted sum of encoder outputs -> the context vector for this step
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights


def build_src_mask(src, pad_id):
    """True where the source token is a real word, False on <pad>."""
    return src != pad_id


if __name__ == "__main__":
    loader, src_vocab, tgt_vocab = build_dataloaders(batch_size=4)
    pad_id = src_vocab.pad_id

    emb_dim, enc_hidden, dec_hidden = 32, 64, 64
    encoder = Encoder(len(src_vocab), emb_dim, enc_hidden, pad_id)
    attention = BahdanauAttention(dec_hidden, enc_hidden)

    src_batch, src_lens, tgt_batch = next(iter(loader))
    enc_outputs, (h, c) = encoder(src_batch, src_lens)
    print(f"encoder outputs: {tuple(enc_outputs.shape)} (batch, src_time, 2*hidden)")
    print(f"bridged decoder init h: {tuple(h.shape)}, c: {tuple(c.shape)}")

    src_mask = build_src_mask(src_batch, pad_id)
    dec_state = h.squeeze(0)  # pretend this is the decoder's current hidden state
    context, weights = attention(dec_state, enc_outputs, src_mask)
    print(f"context: {tuple(context.shape)}, attention weights: {tuple(weights.shape)}")

    # attention weights over real tokens must sum to ~1 per row, and padded
    # positions must carry no weight at all
    row_sums = weights.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    assert torch.all(weights[~src_mask] == 0)
    print("encoder + attention shapes and masking ok")
