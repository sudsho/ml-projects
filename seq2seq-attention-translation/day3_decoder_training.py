"""Day 3 - attention decoder and the teacher-forced training loop.

Days 1-2 left us with padded batches, a bidirectional encoder, and a Bahdanau
attention module. Today we put a decoder on top and actually train the thing.

Three pieces:

  1. AttentionDecoder - one LSTM step at a time. Before each step it queries the
     attention module with its current hidden state, gets a context vector over
     the encoder outputs, and feeds ``[embedding ; context]`` into an LSTMCell.
     The output projection reads ``[hidden ; context ; embedding]`` so the final
     vocabulary distribution sees the aligned source words directly, not just a
     summary buried in the hidden state.

  2. Seq2Seq - the wrapper that runs the encoder once, then unrolls the decoder
     over the target sequence with *teacher forcing*: at step t the decoder is
     fed the ground-truth token from step t-1 rather than its own (early, bad)
     prediction. A scheduled-sampling probability lets us occasionally feed the
     model's own guess so it learns to recover from its mistakes.

  3. The training loop - masked cross-entropy that ignores <pad> targets, Adam,
     and gradient clipping. Unrolled RNNs love to blow the gradient up on a hard
     batch; clipping the global norm keeps a single step from wrecking training.

Day 4 handles greedy/beam decoding, BLEU, and the attention heatmaps.
"""

import random

import torch
import torch.nn as nn

from day1_data import build_dataloaders, SOS, EOS
from day2_encoder_attention import Encoder, BahdanauAttention, build_src_mask


class AttentionDecoder(nn.Module):
    """Single-step attention decoder built on an LSTMCell.

    We drive the decoder one timestep at a time (rather than nn.LSTM over the
    whole sequence) because the attention context feeding each step depends on
    that step's hidden state - there is no way to precompute the inputs for the
    whole sequence up front the way a plain LSTM would want.
    """

    def __init__(self, vocab_size, emb_dim, dec_hidden, enc_hidden, pad_id, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(dec_hidden, enc_hidden)
        # the cell consumes the token embedding concatenated with the context
        self.cell = nn.LSTMCell(emb_dim + 2 * enc_hidden, dec_hidden)
        self.dropout = nn.Dropout(dropout)
        # readout sees hidden state, context, and the input embedding
        self.out = nn.Linear(dec_hidden + 2 * enc_hidden + emb_dim, vocab_size)

    def step(self, token, hidden, cell, enc_outputs, src_mask):
        """Advance one decode step and return logits plus the new state.

        token:   (batch,)                   ids fed at this step
        hidden:  (batch, dec_hidden)        previous decoder hidden state
        returns logits (batch, vocab), new (hidden, cell), attn weights.
        """
        embedded = self.embedding(token)                       # (batch, emb_dim)
        context, weights = self.attention(hidden, enc_outputs, src_mask)
        cell_in = torch.cat([embedded, context], dim=1)
        hidden, cell = self.cell(cell_in, (hidden, cell))
        readout = torch.cat([self.dropout(hidden), context, embedded], dim=1)
        logits = self.out(readout)                             # (batch, vocab)
        return logits, hidden, cell, weights


class Seq2Seq(nn.Module):
    """Encoder + attention decoder with teacher-forced unrolling."""

    def __init__(self, encoder, decoder, pad_id, sos_id):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id
        self.sos_id = sos_id

    def forward(self, src, src_lens, tgt, teacher_forcing=1.0):
        # tgt: (batch, tgt_time) including <sos> ... <eos>. We predict tgt[:, 1:]
        # from tgt[:, :-1], so the loop runs for tgt_time - 1 steps.
        batch, tgt_time = tgt.shape
        vocab_size = self.decoder.out.out_features

        enc_outputs, (h, c) = self.encoder(src, src_lens)
        src_mask = build_src_mask(src, self.pad_id)
        hidden, cell = h.squeeze(0), c.squeeze(0)              # (batch, dec_hidden)

        logits = torch.zeros(batch, tgt_time - 1, vocab_size, device=src.device)
        token = tgt[:, 0]                                      # the <sos> column
        for t in range(1, tgt_time):
            step_logits, hidden, cell, _ = self.decoder.step(
                token, hidden, cell, enc_outputs, src_mask
            )
            logits[:, t - 1] = step_logits
            # teacher forcing: usually feed the gold previous token, but
            # sometimes feed the model's own argmax so it sees its own errors
            if random.random() < teacher_forcing:
                token = tgt[:, t]
            else:
                token = step_logits.argmax(dim=1)
        return logits


def masked_ce_loss(logits, targets, pad_id):
    """Cross-entropy over real target tokens, ignoring <pad> positions.

    ``ignore_index`` drops the padded steps from both the numerator and the
    averaging count, so short sentences in a batch are not penalised for the
    padding that pads them out to the longest sequence.
    """
    vocab = logits.size(-1)
    return nn.functional.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        ignore_index=pad_id,
    )


def train_one_epoch(model, loader, optimizer, pad_id, clip=1.0, teacher_forcing=1.0):
    model.train()
    total_loss, total_batches = 0.0, 0
    for src, src_lens, tgt in loader:
        optimizer.zero_grad()
        logits = model(src, src_lens, tgt, teacher_forcing=teacher_forcing)
        loss = masked_ce_loss(logits, tgt[:, 1:], pad_id)
        loss.backward()
        # clip the global grad norm - unrolled RNN gradients spike on hard
        # batches and one exploding step can undo an epoch of progress
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


def build_model(src_vocab, tgt_vocab, emb_dim=32, enc_hidden=64, dec_hidden=64):
    encoder = Encoder(len(src_vocab), emb_dim, enc_hidden, src_vocab.pad_id)
    decoder = AttentionDecoder(
        len(tgt_vocab), emb_dim, dec_hidden, enc_hidden, tgt_vocab.pad_id
    )
    return Seq2Seq(encoder, decoder, tgt_vocab.pad_id, tgt_vocab.stoi[SOS])


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    loader, src_vocab, tgt_vocab = build_dataloaders(batch_size=4)
    model = build_model(src_vocab, tgt_vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # one forward pass sanity check: logits line up with the shifted target
    src, src_lens, tgt = next(iter(loader))
    logits = model(src, src_lens, tgt)
    assert logits.shape[:2] == (tgt.size(0), tgt.size(1) - 1), logits.shape
    print(f"logits: {tuple(logits.shape)} (batch, tgt_time-1, vocab)")

    # a handful of epochs on the toy corpus should drive the loss down clearly,
    # which tells us the decoder, attention, and masking are all wired together
    first = train_one_epoch(model, loader, optimizer, tgt_vocab.pad_id)
    last = first
    for _ in range(30):
        last = train_one_epoch(model, loader, optimizer, tgt_vocab.pad_id)
    print(f"train loss: {first:.3f} -> {last:.3f}")
    assert last < first, "loss should decrease as the model memorises the toy set"
    print("decoder + teacher-forced training loop ok")
