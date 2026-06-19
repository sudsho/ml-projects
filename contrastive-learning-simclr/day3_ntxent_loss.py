"""Day 3 - the NT-Xent contrastive loss that actually trains SimCLR.

Days 1 and 2 gave us, for a batch of N images, two augmented views each and a
network that maps every view to a normalised projection z. Today we turn those
projections into a learning signal.

The setup. A batch produces 2N projections. View i and its partner i+N come from
the *same* source image, so they are the one positive pair for each other; the
other 2N-2 projections in the batch are negatives. NT-Xent (normalised
temperature-scaled cross entropy) asks, for each anchor, that its positive be
more similar than every negative - it is literally a (2N)-way softmax classifier
whose target is "the partner view".

The pieces:

  - similarity is cosine similarity, so we L2-normalise the projections and take
    dot products. s_{ij} = z_i . z_j.

  - a temperature tau sharpens the distribution. Small tau makes the softmax
    focus hard on the nearest neighbours, which empirically matters a lot; the
    paper uses 0.5 for CIFAR-scale runs.

  - self-similarity s_{ii} must be masked out (an anchor is not its own
    negative), otherwise the trivial "everything is itself" solution leaks in.

For anchor i with positive p(i), the loss is

    L_i = -log( exp(s_{i,p(i)} / tau) / sum_{k != i} exp(s_{i,k} / tau) )

and the batch loss is the mean over all 2N anchors. Below is a from-scratch
implementation plus a small reference using cross_entropy, and a check that the
two agree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """NT-Xent / InfoNCE loss over a batch of 2N projections.

    Expects the two views stacked so that row i and row i+N are a positive pair,
    i.e. z = cat([z_view1, z_view2], dim=0) with z.shape == (2N, dim).
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, z):
        two_n = z.shape[0]
        if two_n % 2 != 0:
            raise ValueError("expected an even number of rows (two views stacked)")
        n = two_n // 2

        # Cosine similarity matrix of every projection against every other.
        z = F.normalize(z, dim=1)
        sim = z @ z.t() / self.temperature                 # (2N, 2N)

        # Mask the diagonal: an anchor cannot be its own negative. Setting it to
        # -inf removes it from the softmax denominator cleanly.
        diag = torch.eye(two_n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(diag, float("-inf"))

        # Target for row i is its partner: i -> i+N and i+N -> i.
        targets = torch.arange(two_n, device=z.device)
        targets = (targets + n) % two_n

        # Each row is now a (2N)-way classification problem over the negatives
        # plus the single positive; cross entropy is exactly NT-Xent.
        return F.cross_entropy(sim, targets)


def ntxent_reference(z, temperature=0.5):
    """Explicit, loop-free reference that builds the loss term by term, used to
    cross-check the vectorised module above."""
    two_n = z.shape[0]
    n = two_n // 2
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temperature

    losses = []
    for i in range(two_n):
        positive = (i + n) % two_n
        # log-sum-exp over all k != i for the denominator.
        denom_terms = [sim[i, k] for k in range(two_n) if k != i]
        denom = torch.logsumexp(torch.stack(denom_terms), dim=0)
        losses.append(-(sim[i, positive] - denom))
    return torch.stack(losses).mean()


if __name__ == "__main__":
    torch.manual_seed(0)

    n, dim = 6, 128
    # Stand-in for day-2 output: two views of N images -> 2N projections.
    z1 = torch.randn(n, dim)
    z2 = z1 + 0.05 * torch.randn(n, dim)        # partner views are correlated
    z = torch.cat([z1, z2], dim=0)

    criterion = NTXentLoss(temperature=0.5)
    loss = criterion(z)
    ref = ntxent_reference(z, temperature=0.5)

    print(f"projections      : {tuple(z.shape)}  (2N rows, dim {dim})")
    print(f"NT-Xent loss     : {loss.item():.6f}")
    print(f"reference loss   : {ref.item():.6f}")
    assert torch.allclose(loss, ref, atol=1e-5), "vectorised vs reference mismatch"

    # Correlated positives should give a lower loss than random projections,
    # a quick sanity check that the loss rewards agreement between views.
    z_rand = torch.randn(2 * n, dim)
    rand_loss = criterion(z_rand)
    print(f"random-pair loss : {rand_loss.item():.6f}  (should exceed {loss.item():.6f})")
    assert rand_loss > loss

    # Lower temperature sharpens the contrast and changes the scale of the loss.
    sharp = NTXentLoss(temperature=0.1)(z)
    print(f"loss at tau=0.1  : {sharp.item():.6f}")
    print("NT-Xent loss ready - day 4 wires it into the training loop + linear probe")
