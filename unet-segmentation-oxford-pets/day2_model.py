"""Day 2 - the U-Net architecture from scratch.

Day 1 gave us image/mask batches at 128x128 with three label classes. Today we
build the network that turns an image tensor into a per-pixel class score map of
the same spatial size. U-Net is an encoder/decoder: the encoder repeatedly halves
the resolution while doubling the channel count (capturing "what" is in the image
at the cost of "where"), and the decoder upsamples back to full resolution. The
trick that makes it work for segmentation is the skip connection - at each decoder
step we concatenate the matching-resolution encoder feature map, handing the fine
spatial detail lost during downsampling directly to the upsampling path.

The building block is a "double conv": two 3x3 convolutions each followed by batch
norm and ReLU. Encoder stages are double-conv then 2x2 max-pool; decoder stages are
a 2x2 transposed conv (learned upsample), concatenation with the skip, then another
double conv. A final 1x1 conv projects to NUM_CLASSES logits. Days 3-4 add the
Dice + cross-entropy loss and training loop, then evaluation and visualization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from day1_data import IMAGE_SIZE, NUM_CLASSES


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x2, the repeated unit of every U-Net stage.

    Padding of 1 keeps the spatial size unchanged so the skip-connection tensors
    line up exactly with their decoder counterparts. Batch norm was not in the
    original 2015 paper but stabilises training from scratch noticeably here.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """U-Net with a configurable channel schedule.

    The encoder stores each stage's output before pooling so the decoder can
    concatenate it back. The bottleneck sits at the lowest resolution with the most
    channels. Each decoder stage upsamples, pads if an odd size ever leaves a
    one-pixel mismatch, concatenates the stored skip, then double-convs.
    """

    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, base=64):
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8]  # 64, 128, 256, 512

        # Encoder: a double conv per stage; pooling is shared and stateless.
        self.encoders = nn.ModuleList()
        prev = in_channels
        for w in widths:
            self.encoders.append(DoubleConv(prev, w))
            prev = w
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(widths[-1], widths[-1] * 2)  # 512 -> 1024

        # Decoder: for each stage, a transposed conv that halves channels and
        # doubles resolution, followed by a double conv over the concatenation.
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = widths[-1] * 2
        for w in reversed(widths):
            self.upconvs.append(nn.ConvTranspose2d(prev, w, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(w * 2, w))  # w (skip) + w (upsampled)
            prev = w

        self.head = nn.Conv2d(widths[0], num_classes, kernel_size=1)

    def forward(self, x):
        """Map an image batch [B, in_channels, H, W] to logits [B, num_classes, H, W].

        The encoder loop caches each stage output as a skip before pooling; the
        decoder loop consumes those skips in reverse, upsampling and concatenating
        at each step, and the 1x1 head projects to per-pixel class scores.
        """
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # Guard against off-by-one spatial mismatches from odd input sizes.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.head(x)


def count_parameters(model):
    """Number of trainable parameters - a quick sanity check on model size."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=NUM_CLASSES, base=64)
    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    logits = model(dummy)
    print(f"input      : {tuple(dummy.shape)}")
    print(f"output     : {tuple(logits.shape)}  (expect [2, {NUM_CLASSES}, {IMAGE_SIZE}, {IMAGE_SIZE}])")
    assert logits.shape == (2, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)
    print(f"parameters : {count_parameters(model):,}")
