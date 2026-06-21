"""Day 1 - data pipeline and the DCGAN generator/discriminator for Fashion-MNIST.

This project builds a Deep Convolutional GAN from scratch in PyTorch and trains
it to generate 28x28 Fashion-MNIST images. Unlike the VAE and diffusion projects
in this repo, a GAN has no explicit likelihood: a generator turns a noise vector
into an image, a discriminator tries to tell real images from generated ones, and
the two are trained against each other in a min-max game. There is no encoder and
no reconstruction term - the only learning signal is the discriminator's verdict.

The DCGAN paper (Radford et al., 2015) is really a set of architectural rules that
make that adversarial game stable enough to train with plain convolutions:

  - replace pooling with strided convolutions (discriminator) and fractionally
    strided / transposed convolutions (generator) so the nets learn their own
    spatial down/up-sampling,
  - use batch norm in both nets (except the generator output and discriminator
    input layers, which would otherwise have their statistics distorted),
  - ReLU in the generator (Tanh on the output), LeakyReLU in the discriminator,
  - no fully connected hidden layers.

Fashion-MNIST is single channel at 28x28. To keep the transposed-conv arithmetic
clean we generate at 32x32 and center-crop/resize the reals to match, a common
trick so the spatial sizes are powers of two. Days 2-4 add the loss + train step,
the full adversarial training loop, and sampling/interpolation + the README.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader

# Generator input noise dimensionality and the base feature-map width that the
# channel counts are multiples of. 64 is the DCGAN default and plenty for 32x32.
LATENT_DIM = 100
FEATURE_MAPS = 64
IMAGE_SIZE = 32
CHANNELS = 1


def get_dataloader(batch_size=128, root="./data", train=True):
    """Fashion-MNIST loader normalised to [-1, 1] to match the generator's Tanh.

    Images are resized 28 -> 32 so every spatial dimension is a power of two,
    which lets the generator reach 32x32 with exactly four transposed-conv
    doublings (4 -> 8 -> 16 -> 32) and the discriminator mirror it back down.
    Normalising to [-1, 1] (rather than [0, 1]) is important: the generator's
    final activation is Tanh, so the real data must live in the same range or the
    discriminator gets a trivial range-based shortcut to separate the two.
    """
    transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1,1]
    ])
    dataset = datasets.FashionMNIST(
        root=root, train=train, download=True, transform=transform
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=2
    )


def weights_init(module):
    """DCGAN weight init: N(0, 0.02) for conv weights, BN gamma near 1.

    The paper initialises all weights from a zero-mean normal with std 0.02.
    Without this GANs frequently collapse early because one network overpowers
    the other before any useful gradients flow. BatchNorm scale is centred at 1
    with the same small jitter and its bias zeroed.
    """
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class Generator(nn.Module):
    """Maps a latent vector z (LATENT_DIM,) to a 1x32x32 image in [-1, 1].

    The noise vector is treated as a 1x1 "image" with LATENT_DIM channels, then
    four transposed convolutions progressively trade channels for spatial size:
    1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32, halving the channel count at each step.
    BatchNorm + ReLU follow every hidden layer; the output layer uses Tanh and no
    norm so the network can freely hit the [-1, 1] extremes.
    """

    def __init__(self, latent_dim=LATENT_DIM, fmaps=FEATURE_MAPS, channels=CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            # z: (latent_dim, 1, 1) -> (fmaps*4, 4, 4)
            nn.ConvTranspose2d(latent_dim, fmaps * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fmaps * 4),
            nn.ReLU(True),
            # -> (fmaps*2, 8, 8)
            nn.ConvTranspose2d(fmaps * 4, fmaps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmaps * 2),
            nn.ReLU(True),
            # -> (fmaps, 16, 16)
            nn.ConvTranspose2d(fmaps * 2, fmaps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmaps),
            nn.ReLU(True),
            # -> (channels, 32, 32)
            nn.ConvTranspose2d(fmaps, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        # accept (N, latent_dim) and reshape to the 1x1 spatial map the convs want
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    """Maps a 1x32x32 image to a single real/fake logit.

    A mirror of the generator: strided convolutions shrink 32 -> 16 -> 8 -> 4
    while growing channels, then a final conv collapses the 4x4 map to one scalar
    logit (no sigmoid here - we use BCEWithLogits in day 2 for numerical safety).
    LeakyReLU keeps a small gradient on the negative side, which the paper found
    important for the discriminator. The first layer has no BatchNorm so the input
    image statistics are not washed out before any features are extracted.
    """

    def __init__(self, fmaps=FEATURE_MAPS, channels=CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            # (channels, 32, 32) -> (fmaps, 16, 16)
            nn.Conv2d(channels, fmaps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (fmaps*2, 8, 8)
            nn.Conv2d(fmaps, fmaps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmaps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (fmaps*4, 4, 4)
            nn.Conv2d(fmaps * 2, fmaps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmaps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (1, 1, 1) logit
            nn.Conv2d(fmaps * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)  # flatten to (N,) logits


def _sanity_check():
    """Smoke test: shapes line up and a noise batch flows G -> D end to end."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    z = torch.randn(8, LATENT_DIM, device=device)
    fake = netG(z)
    assert fake.shape == (8, CHANNELS, IMAGE_SIZE, IMAGE_SIZE), fake.shape
    assert fake.min() >= -1.0 and fake.max() <= 1.0

    logits = netD(fake)
    assert logits.shape == (8,), logits.shape

    g_params = sum(p.numel() for p in netG.parameters())
    d_params = sum(p.numel() for p in netD.parameters())
    print(f"generator output: {tuple(fake.shape)}, range "
          f"[{fake.min():.2f}, {fake.max():.2f}]")
    print(f"discriminator logits: {tuple(logits.shape)}")
    print(f"params -> G: {g_params/1e6:.2f}M, D: {d_params/1e6:.2f}M")


if __name__ == "__main__":
    _sanity_check()
