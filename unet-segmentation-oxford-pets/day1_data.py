"""Day 1 - the Oxford-IIIT Pet data pipeline for semantic segmentation.

This project builds a U-Net from scratch in PyTorch and trains it to segment the
Oxford-IIIT Pet dataset pixel by pixel. Unlike the classification projects in this
repo (ViT, the CNN on CIFAR), the label here is not a single class per image but a
full-resolution mask assigning every pixel to one of a few classes. That changes
the whole data pipeline: any spatial transform applied to the image - resize, crop,
flip - has to be applied identically to the mask, or the pixels stop lining up.

Oxford-IIIT Pet ships a "trimap" annotation per image where each pixel is:

  1 = foreground (the animal),
  2 = background,
  3 = boundary / not-classified (the fuzzy outline around the animal).

torchvision's OxfordIIITPet with target_types="segmentation" returns those trimaps
as PIL images with pixel values {1, 2, 3}. We remap them to a clean 0-based label
space {0, 1, 2} = {background, foreground, boundary} so they can be fed straight to
cross-entropy on day 3. Days 2-4 add the U-Net itself, the Dice + cross-entropy
loss and training loop with mean-IoU, then mask visualization and the README.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

# Everything is resized to a fixed square so the encoder's four 2x downsamples
# divide evenly (128 -> 64 -> 32 -> 16 -> 8) and masks can be batched together.
IMAGE_SIZE = 128
NUM_CLASSES = 3  # background, foreground, boundary
# ImageNet statistics - the encoder is trained from scratch here, but normalising
# to these keeps the input distribution well-conditioned and lets us swap in a
# pretrained backbone later without touching the pipeline.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class JointTransform:
    """Apply the SAME geometric transform to an image and its segmentation mask.

    The image and mask must stay pixel-aligned, so random spatial augmentation
    cannot be done with two independent transform pipelines. Here we sample the
    augmentation parameters once and apply them to both: resize with bilinear for
    the image but nearest for the mask (bilinear would invent fractional class
    ids), an optional shared horizontal flip, then image-only colour/normalise.
    """

    def __init__(self, image_size=IMAGE_SIZE, train=True):
        self.image_size = image_size
        self.train = train
        self.color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2)

    def __call__(self, image, mask):
        # Resize both - bilinear keeps the photo smooth, nearest keeps mask labels
        # integer-valued so no spurious class ids appear at the interpolated edges.
        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(
            mask, [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )

        if self.train and torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if self.train:
            image = self.color_jitter(image)

        image = TF.to_tensor(image)
        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)

        # Trimap {1,2,3} -> labels {0,1,2}; store as long for cross-entropy.
        mask = torch.from_numpy(np.array(mask, dtype=np.int64)) - 1
        mask = mask.clamp(0, NUM_CLASSES - 1)
        return image, mask


class PetSegmentation(torch.utils.data.Dataset):
    """Thin wrapper that routes the raw (PIL image, PIL trimap) pair through the
    joint transform so image and mask are augmented together."""

    def __init__(self, root="./data", split="trainval", train=True):
        self.base = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=True,
        )
        self.joint = JointTransform(train=train)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, mask = self.base[idx]  # PIL RGB, PIL trimap
        return self.joint(image, mask)


def get_dataloaders(root="./data", batch_size=16, val_fraction=0.15, seed=42):
    """Build train/val loaders from the Oxford-IIIT Pet trainval split.

    The dataset has no official validation split for segmentation, so we carve one
    out of trainval with a fixed generator seed for reproducibility. The training
    subset keeps augmentation on; the validation subset uses the deterministic
    transform (no flips, no jitter) so metrics are comparable across epochs.
    """
    full_train = PetSegmentation(root=root, split="trainval", train=True)
    n_val = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)
    # Point the validation subset at a non-augmenting transform for clean metrics.
    val_set.dataset = PetSegmentation(root=root, split="trainval", train=False)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader


def class_pixel_frequencies(loader, num_batches=20):
    """Estimate the per-class pixel share over a few batches.

    Pet masks are heavily imbalanced - background dominates and the boundary class
    is a thin ring of pixels. Knowing the split up front motivates the Dice term
    added on day 3, which is far less sensitive to that imbalance than plain
    cross-entropy weighted by pixel count.
    """
    counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
    for i, (_, masks) in enumerate(loader):
        if i >= num_batches:
            break
        counts += torch.bincount(masks.flatten(), minlength=NUM_CLASSES)
    total = counts.sum().item()
    return {c: counts[c].item() / total for c in range(NUM_CLASSES)}


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=8)
    images, masks = next(iter(train_loader))
    print(f"image batch : {tuple(images.shape)}  dtype={images.dtype}")
    print(f"mask batch  : {tuple(masks.shape)}  dtype={masks.dtype}")
    print(f"mask labels : {sorted(masks.unique().tolist())}")
    print(f"train / val : {len(train_loader.dataset)} / {len(val_loader.dataset)}")
    print(f"class share : {class_pixel_frequencies(train_loader)}")
