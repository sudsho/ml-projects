"""Day 4 - greedy and beam-search decoding, BLEU, attention heatmaps.

Days 1-3 built the data pipeline, the bidirectional encoder with Bahdanau
attention, and a teacher-forced training loop. Training always had the gold
previous token to lean on; inference does not. Today we generate translations
for real and measure how good they are.

Four pieces:

  1. Greedy decoding - run the encoder once, then unroll the decoder feeding its
     own argmax back in until it emits <eos> (or we hit a length cap). We collect
     the attention weight vector at every step so we can draw the alignment later.

  2. Beam search - keep the ``k`` best partial hypotheses instead of committing to
     the single argmax at each step. Scoring by summed log-probability biases
     toward short sentences, so we length-normalise before the final pick. Greedy
     is just beam search with k=1, and on the toy corpus the two usually agree,
     but beam is the version that matters once the vocabulary grows.

  3. BLEU - modified n-gram precision with the brevity penalty. Modified precision
     clips each predicted n-gram count by how often it actually appears in the
     reference, which stops a model from spamming one common word. The brevity
     penalty punishes translations that are too short to game precision.

  4. Attention heatmaps - the (target, source) weight matrix greedy decoding hands
     back, drawn as an image so you can eyeball whether the alignments make sense.
"""

import math
from collections import Counter

import torch

from day1_data import build_dataloaders, SOS, EOS
from day2_encoder_attention import build_src_mask
from day3_decoder_training import build_model, train_one_epoch


@torch.no_grad()
def greedy_decode(model, src, src_lens, sos_id, eos_id, max_len=20):
    """Translate a single source sentence, feeding predictions back in.

    Returns the list of predicted token ids (without <sos>, cut at <eos>) and the
    stacked attention weights of shape (steps, src_time) for the heatmap.
    """
    model.eval()
    enc_outputs, (h, c) = model.encoder(src, src_lens)
    src_mask = build_src_mask(src, model.pad_id)
    hidden, cell = h.squeeze(0), c.squeeze(0)

    token = torch.full((src.size(0),), sos_id, dtype=torch.long, device=src.device)
    tokens, attentions = [], []
    for _ in range(max_len):
        logits, hidden, cell, weights = model.decoder.step(
            token, hidden, cell, enc_outputs, src_mask
        )
        token = logits.argmax(dim=1)
        attentions.append(weights.squeeze(0))
        if token.item() == eos_id:
            break
        tokens.append(token.item())
    attn = torch.stack(attentions) if attentions else torch.empty(0)
    return tokens, attn


@torch.no_grad()
def beam_search_decode(model, src, src_lens, sos_id, eos_id, beam_size=3, max_len=20, alpha=0.7):
    """Beam search over a single sentence with length-normalised scoring.

    Each live hypothesis carries its token list, running log-prob, and decoder
    state. We expand every beam, keep the ``beam_size`` best continuations
    overall, and retire a hypothesis once it emits <eos>. The winner is the
    finished hypothesis with the best length-normalised score.
    """
    model.eval()
    enc_outputs, (h, c) = model.encoder(src, src_lens)
    src_mask = build_src_mask(src, model.pad_id)

    # a beam entry: (tokens, logprob, hidden, cell)
    beams = [([], 0.0, h.squeeze(0), c.squeeze(0))]
    finished = []
    for _ in range(max_len):
        candidates = []
        for tokens, logprob, hidden, cell in beams:
            last = tokens[-1] if tokens else sos_id
            token = torch.tensor([last], device=src.device)
            logits, new_h, new_c, _ = model.decoder.step(
                token, hidden, cell, enc_outputs, src_mask
            )
            log_probs = torch.log_softmax(logits.squeeze(0), dim=-1)
            topv, topi = log_probs.topk(beam_size)
            for v, i in zip(topv.tolist(), topi.tolist()):
                candidates.append((tokens + [i], logprob + v, new_h, new_c))

        # keep the globally best continuations, then peel off any that hit <eos>
        candidates.sort(key=lambda b: b[1], reverse=True)
        beams = []
        for cand in candidates:
            if cand[0][-1] == eos_id:
                finished.append(cand)
            else:
                beams.append(cand)
            if len(beams) == beam_size:
                break
        if not beams:
            break

    pool = finished if finished else beams
    # length normalisation: divide by length^alpha so beam search does not just
    # prefer the shortest sequence with the fewest negative log-probs summed in
    best = max(pool, key=lambda b: b[1] / (max(len(b[0]), 1) ** alpha))
    tokens = best[0]
    if tokens and tokens[-1] == eos_id:
        tokens = tokens[:-1]
    return tokens


def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hypotheses, references, max_n=4):
    """Corpus-level BLEU with the brevity penalty and add-1 smoothing.

    hypotheses / references are lists of token lists (one reference per hypothesis
    here for simplicity). We accumulate clipped n-gram matches across the whole
    corpus before dividing, which is what makes corpus BLEU steadier than
    averaging per-sentence scores.
    """
    clipped = [0] * max_n
    totals = [0] * max_n
    hyp_len, ref_len = 0, 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            hyp_ngrams = _ngram_counts(hyp, n)
            ref_ngrams = _ngram_counts(ref, n)
            overlap = sum(min(c, ref_ngrams[g]) for g, c in hyp_ngrams.items())
            clipped[n - 1] += overlap
            totals[n - 1] += max(len(hyp) - n + 1, 0)

    # add-1 smoothing keeps a single missing higher-order n-gram from zeroing BLEU
    precisions = []
    for n in range(max_n):
        num = clipped[n] + (1 if n > 0 else 0)
        den = totals[n] + (1 if n > 0 else 0)
        precisions.append(num / den if den > 0 else 0.0)

    if min(precisions) == 0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)

    # brevity penalty: no reward for being terse, but do not penalise long output
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return bp * geo_mean


def plot_attention(attn, src_tokens, pred_tokens, path):
    """Save the (target, source) attention matrix as a heatmap, if matplotlib is
    available. Rows are generated target words, columns are the source words the
    decoder was attending to at that step."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping heatmap")
        return

    fig, ax = plt.subplots(figsize=(max(len(src_tokens), 4), max(len(pred_tokens), 4)))
    ax.imshow(attn[: len(pred_tokens)].cpu().numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(pred_tokens)))
    ax.set_yticklabels(pred_tokens)
    ax.set_xlabel("source")
    ax.set_ylabel("prediction")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved attention heatmap to {path}")


if __name__ == "__main__":
    torch.manual_seed(0)

    loader, src_vocab, tgt_vocab = build_dataloaders(batch_size=4)
    model = build_model(src_vocab, tgt_vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # overfit the toy corpus so decoding has something coherent to produce
    for _ in range(60):
        train_one_epoch(model, loader, optimizer, tgt_vocab.pad_id)

    sos_id, eos_id = tgt_vocab.stoi[SOS], tgt_vocab.stoi[EOS]
    hyps, refs = [], []
    for src, src_lens, tgt in loader:
        for b in range(src.size(0)):
            one_len = src_lens[b : b + 1]
            # trim to the true length so the attention mask width lines up with
            # the encoder outputs (packing collapses to this sentence's length)
            one_src = src[b : b + 1, : one_len.item()]

            greedy_ids, attn = greedy_decode(model, one_src, one_len, sos_id, eos_id)
            beam_ids = beam_search_decode(model, one_src, one_len, sos_id, eos_id)

            src_tokens = src_vocab.decode(one_src.squeeze(0).tolist())
            gold = tgt_vocab.decode(tgt[b].tolist())
            greedy_tokens = [tgt_vocab.itos[i] for i in greedy_ids]
            beam_tokens = [tgt_vocab.itos[i] for i in beam_ids]

            hyps.append(greedy_tokens)
            refs.append(gold)
            print(f"src   : {' '.join(src_tokens)}")
            print(f"gold  : {' '.join(gold)}")
            print(f"greedy: {' '.join(greedy_tokens)}")
            print(f"beam  : {' '.join(beam_tokens)}")
            print("-" * 40)

    bleu = corpus_bleu(hyps, refs)
    print(f"corpus BLEU on the toy set: {bleu:.3f}")

    # draw the alignment for the first sentence so day 4 leaves an artefact behind
    src, src_lens, _ = next(iter(loader))
    first_src = src[:1, : src_lens[0].item()]
    greedy_ids, attn = greedy_decode(model, first_src, src_lens[:1], sos_id, eos_id)
    if greedy_ids:
        src_tokens = src_vocab.decode(first_src.squeeze(0).tolist())
        pred_tokens = [tgt_vocab.itos[i] for i in greedy_ids]
        plot_attention(attn, src_tokens, pred_tokens, "attention_heatmap.png")

    print("decoding + BLEU + heatmap ok")
