"""Day 1 - the stochastic augmentation pipeline at the heart of SimCLR.

SimCLR learns representations with no labels at all. The trick is to treat two
randomly augmented views of the *same* image as a positive pair, and every other
image in the batch as a negative. The encoder is then pushed to map the two views
close together in embedding space while spreading different images apart.

Everything therefore hinges on the augmentations: they have to distort an image
hard enough that matching its two views is non-trivial, but not so hard that the
underlying object becomes unrecognisable. The SimCLR paper found the strongest
combination to be random resized crop + color jitter + random grayscale + (for
larger images) Gaussian blur, applied independently to produce each view.

This module builds that pipeline and a small wrapper dataset that returns the two
correlated views per image. Days 2-4 add the encoder + projection head, the
NT-Xent loss, and the linear-probe evaluation.
"""

import torch
import torchvision.transforms as T
from torchvision import datasets

# CIFAR-10 channel statistics, used to normalise inputs to the encoder.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class GaussianBlur:
    """Light Gaussian blur applied with probability `p`.

    The paper blurs only a fraction of the time so the encoder still sees plenty
    of sharp images. Kernel size is taken as ~10% of the image side (forced odd),
    matching the recipe, with sigma drawn uniformly from a small range.
    """

    def __init__(self, image_size, p=0.5, sigma=(0.1, 2.0)):
        kernel = int(0.1 * image_size)
        if kernel % 2 == 0:
            kernel += 1
        self.blur = T.GaussianBlur(kernel_size=kernel, sigma=sigma)
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            return self.blur(img)
        return img


def build_simclr_transform(image_size=32, blur=False, color_jitter_strength=0.5):
    """Compose the stochastic augmentation used to generate a single view.

    The components, in order, are:
      - random resized crop back to `image_size` (scale-and-aspect distortion),
      - random horizontal flip,
      - color jitter applied 80% of the time (brightness/contrast/saturation/hue),
      - random grayscale 20% of the time (removes the color shortcut),
      - optional Gaussian blur (helpful for larger images, off by default on 32px),
      - tensor conversion and channel normalisation.

    `color_jitter_strength` scales the jitter magnitudes together, as `s` does in
    the paper; hue gets a quarter of the strength since it is far more disruptive.
    """
    s = color_jitter_strength
    jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

    steps = [
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([jitter], p=0.8),
        T.RandomGrayscale(p=0.2),
    ]
    if blur:
        steps.append(GaussianBlur(image_size, p=0.5))
    steps += [
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return T.Compose(steps)


class TwoViewTransform:
    """Apply the same stochastic transform twice to get two correlated views.

    Because the transform is random, calling it twice on one image yields two
    different augmentations - exactly the positive pair SimCLR contrasts against
    the rest of the batch.
    """

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, img):
        return self.base_transform(img), self.base_transform(img)


def build_contrastive_dataset(root="./data", train=True, image_size=32, blur=False):
    """Return a CIFAR-10 dataset whose items are (view_1, view_2) tensor pairs."""
    transform = TwoViewTransform(build_simclr_transform(image_size, blur=blur))
    return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)


def collate_views(batch):
    """Collate (view1, view2) pairs into two stacked tensors of shape (B, C, H, W).

    Returning the views as separate batches keeps the day-3 loss code simple: it
    can concatenate them itself and know the first B rows pair with the next B.
    """
    view1 = torch.stack([item[0][0] for item in batch])
    view2 = torch.stack([item[0][1] for item in batch])
    return view1, view2


if __name__ == "__main__":
    torch.manual_seed(0)

    # Build the pipeline and inspect a single image's two views.
    transform = build_simclr_transform(image_size=32)
    two_view = TwoViewTransform(transform)

    dummy = torch.rand(3, 32, 32)  # stand-in for a PIL image when run offline
    pil_transform = T.ToPILImage()
    v1, v2 = two_view(pil_transform(dummy))

    print(f"view 1 shape    : {tuple(v1.shape)}")
    print(f"view 2 shape    : {tuple(v2.shape)}")
    print(f"views identical : {torch.equal(v1, v2)}")  # expect False - random aug
    print(f"view 1 mean/std : {v1.mean():.3f} / {v1.std():.3f}")

    # Sanity-check the batched collate path on a few fake samples.
    fake_batch = [((v1, v2), 0) for _ in range(8)]
    b1, b2 = collate_views(fake_batch)
    print(f"batched views   : {tuple(b1.shape)} and {tuple(b2.shape)}")
    assert b1.shape == b2.shape == (8, 3, 32, 32)
    print("augmentation pipeline ready - two correlated views per image")
