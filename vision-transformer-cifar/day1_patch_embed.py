"""Day 1 - CIFAR-10 pipeline and the ViT patch-embedding front end.

This project builds a Vision Transformer (Dosovitskiy et al., 2020) from scratch
in PyTorch and trains it to classify CIFAR-10. The repo already has a decoder-only
transformer for text (char-transformer-shakespeare); a ViT is the mirror image of
that idea applied to images. The core insight of the paper is almost provocative:
once you chop an image into a grid of fixed-size patches and treat each patch as a
token, a *standard* transformer encoder - the same machinery used for language -
can classify images with no convolutions at all.

The pieces that turn an image into a token sequence, all of which live in this
file, are:

  - Patchify. Split a 32x32x3 image into non-overlapping P x P patches. With P=4
    that is an 8x8 grid = 64 patches, each flattened to a 4*4*3 = 48-dim vector.
  - Linear projection. Map every flat patch to the model width D with a single
    Linear layer. Equivalently this is a Conv2d with kernel=stride=P, which is the
    trick we use here because it does the split-and-project in one fused op.
  - Class token. Prepend one learnable [CLS] vector to the sequence; the encoder's
    final state at this position is what the classifier head reads. The token has
    no spatial meaning of its own - it is a slot that aggregates global info.
  - Positional embeddings. Self-attention is permutation invariant, so without a
    position signal the model cannot tell a patch's top-left from its bottom-right.
    We add a learnable embedding (one per sequence position, CLS included).

Days 2-4 add multi-head self-attention and the encoder block, assemble the full
model with a training loop, then evaluate, visualize attention, and write the
project README.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader

# CIFAR-10 is 32x32x3. A patch size of 4 gives an 8x8 = 64-token sequence, which
# is small enough to train on a single GPU yet long enough for attention to have
# something to do. D=192 / 3 heads is the "ViT-Tiny" width - sane for 32x32.
IMAGE_SIZE = 32
PATCH_SIZE = 4
CHANNELS = 3
EMBED_DIM = 192

# CIFAR-10 channel statistics, used to standardise inputs.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(batch_size=128, root="./data"):
    """CIFAR-10 train/test loaders with light augmentation on the train split.

    ViTs have far less built-in inductive bias than CNNs (no locality, no
    translation equivariance baked into the weights), so on a small dataset like
    CIFAR-10 they lean heavily on augmentation to avoid overfitting. We keep it
    standard for day 1: random crop with reflection padding and a horizontal flip.
    Test data is only normalised.
    """
    train_tf = T.Compose([
        T.RandomCrop(IMAGE_SIZE, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=2)
    return train_loader, test_loader


class PatchEmbedding(nn.Module):
    """Turn an image batch into a sequence of patch tokens with CLS + position.

    The split-into-patches-then-project step is implemented as a single strided
    convolution: a Conv2d with kernel_size = stride = patch_size sees each patch
    exactly once and projects it to EMBED_DIM, which is mathematically identical
    to flattening the patch and applying a Linear layer but far cleaner to write.
    """

    def __init__(self, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
                 channels=CHANNELS, embed_dim=EMBED_DIM):
        super().__init__()
        assert image_size % patch_size == 0, "image size must be divisible by patch size"
        self.num_patches = (image_size // patch_size) ** 2

        # kernel = stride = patch_size => non-overlapping patches, one conv = one
        # linear projection per patch.
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

        # One learnable CLS token, broadcast across the batch at forward time.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable absolute positions for every token, CLS included (+1).
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Truncated-normal init mirrors the reference ViT setup.
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, num_patches, D)
        b = x.shape[0]
        x = self.proj(x)                      # (B, D, gh, gw)
        x = x.flatten(2).transpose(1, 2)      # (B, num_patches, D)

        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, D)
        x = torch.cat((cls, x), dim=1)           # (B, num_patches + 1, D)
        x = x + self.pos_embed                   # add position to every token
        return x


if __name__ == "__main__":
    # Shape sanity check without needing the dataset downloaded: a random batch
    # should come out as (B, num_patches + 1, EMBED_DIM).
    embed = PatchEmbedding()
    dummy = torch.randn(8, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    tokens = embed(dummy)
    expected = (8, (IMAGE_SIZE // PATCH_SIZE) ** 2 + 1, EMBED_DIM)
    print("patch grid:", IMAGE_SIZE // PATCH_SIZE, "x", IMAGE_SIZE // PATCH_SIZE)
    print("token sequence:", tuple(tokens.shape), "expected", expected)
    assert tuple(tokens.shape) == expected
    print("ok")
