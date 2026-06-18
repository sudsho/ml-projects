"""Day 2 - the encoder and projection head SimCLR contrasts in embedding space.

Day 1 produced two correlated views per image. Today we build the network that
turns a view into a vector. SimCLR splits this into two parts:

  - an *encoder* f(.) - a standard CNN backbone (here a CIFAR-friendly ResNet)
    whose penultimate features h = f(x) are what we ultimately keep and probe
    with a linear classifier in day 4;

  - a *projection head* g(.) - a small MLP that maps h to the lower-dimensional
    space z = g(h) where the contrastive (NT-Xent) loss is actually applied.

The paper's key empirical finding is that contrasting on z rather than directly
on h gives much better downstream features: the head absorbs the information that
is useful for the contrastive task but not for classification, so h is left
cleaner. After pre-training the head is thrown away and only the encoder is kept.

The stock torchvision ResNet starts with a 7x7 stride-2 conv and a max-pool,
which is far too aggressive for 32x32 CIFAR images - it would collapse them to a
couple of pixels before any real processing. We therefore swap in a 3x3 stride-1
stem and drop the initial pooling, the standard CIFAR ResNet modification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """The two-conv residual block used by ResNet-18/34."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Projection shortcut when shape changes, identity otherwise.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CIFARResNetEncoder(nn.Module):
    """ResNet backbone with a CIFAR stem, returning pooled features h.

    `layers` gives the number of residual blocks per stage; [2, 2, 2, 2] is
    ResNet-18. The final global-average-pooled vector has `feature_dim`
    dimensions (512 for the BasicBlock variants) and is what day 4 linearly
    probes - the classification head is deliberately absent.
    """

    def __init__(self, layers=(2, 2, 2, 2)):
        super().__init__()
        self.in_planes = 64

        # CIFAR stem: 3x3 stride-1, no max-pool, to preserve spatial resolution.
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_stage(64, layers[0], stride=1)
        self.layer2 = self._make_stage(128, layers[1], stride=2)
        self.layer3 = self._make_stage(256, layers[2], stride=2)
        self.layer4 = self._make_stage(512, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 512 * BasicBlock.expansion

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        return torch.flatten(out, 1)  # (B, feature_dim)


class ProjectionHead(nn.Module):
    """Two-layer MLP mapping encoder features h to the contrastive space z.

    SimCLR uses a single hidden layer with ReLU and BatchNorm. The output z is
    L2-normalised in the loss (day 3) so that cosine similarity reduces to a dot
    product; we leave normalisation out here to keep the module loss-agnostic.
    """

    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h):
        return self.net(h)


class SimCLRModel(nn.Module):
    """Encoder + projection head. Returns (h, z); day 4 keeps only the encoder."""

    def __init__(self, layers=(2, 2, 2, 2), proj_hidden=512, proj_out=128):
        super().__init__()
        self.encoder = CIFARResNetEncoder(layers)
        self.projector = ProjectionHead(self.encoder.feature_dim, proj_hidden, proj_out)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


def count_parameters(module):
    """Trainable-parameter count, handy for sanity-checking model size."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)

    model = SimCLRModel()
    # A fake batch of 8 CIFAR-sized views, standing in for day-1 output.
    x = torch.randn(8, 3, 32, 32)
    h, z = model(x)

    print(f"input            : {tuple(x.shape)}")
    print(f"encoder feature h: {tuple(h.shape)}   (kept for the linear probe)")
    print(f"projection z     : {tuple(z.shape)}   (where NT-Xent is applied)")
    assert h.shape == (8, 512)
    assert z.shape == (8, 128)

    print(f"encoder params   : {count_parameters(model.encoder):,}")
    print(f"projector params : {count_parameters(model.projector):,}")

    # Cosine similarity of two normalised projections - a preview of day 3, where
    # the positive pair's similarity is pushed up against all in-batch negatives.
    z_norm = F.normalize(z, dim=1)
    sim = z_norm @ z_norm.t()
    print(f"similarity matrix: {tuple(sim.shape)}  diag mean {sim.diag().mean():.3f}")
    print("encoder + projection head ready - day 3 adds the NT-Xent loss")
