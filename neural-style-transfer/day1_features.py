"""
Day 1 of neural style transfer.

Sets up the pretrained VGG19 backbone, the image preprocessing pipeline, and
a feature extractor that pulls activations from the conv layers we care about
for content and style. We also do a quick sanity check on a content/style pair
to confirm the activation shapes line up with what Gatys et al. used.
"""

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# Layer indices follow the original Gatys et al. (2015) paper. VGG19's features
# module is indexed 0..36 and these are the conv outputs (post-ReLU) at the
# beginning of each block.
CONTENT_LAYERS = ("conv4_2",)
STYLE_LAYERS = ("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")

VGG19_LAYER_MAP = {
    "conv1_1": 0,
    "conv2_1": 5,
    "conv3_1": 10,
    "conv4_1": 19,
    "conv4_2": 21,
    "conv5_1": 28,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_preprocess(image_size=512):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_image(path, transform, device):
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return tensor.to(device)


class VGGFeatureExtractor(nn.Module):
    """Pulls intermediate activations out of VGG19 by name."""

    def __init__(self, layer_names):
        super().__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1
        vgg = models.vgg19(weights=weights).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)

        max_idx = max(VGG19_LAYER_MAP[name] for name in layer_names)
        self.vgg = vgg[: max_idx + 1]
        self.layer_indices = OrderedDict(
            (name, VGG19_LAYER_MAP[name]) for name in layer_names
        )

    def forward(self, x):
        outputs = OrderedDict()
        wanted = set(self.layer_indices.values())
        index_to_name = {v: k for k, v in self.layer_indices.items()}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in wanted:
                outputs[index_to_name[i]] = x
            if i == max(wanted):
                break
        return outputs


def report_shapes(features):
    for name, tensor in features.items():
        c, h, w = tensor.shape[1:]
        print(f"  {name}: channels={c} h={h} w={w}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    preprocess = build_preprocess(image_size=512)
    extractor = VGGFeatureExtractor(CONTENT_LAYERS + STYLE_LAYERS).to(device)

    samples = Path("samples")
    if samples.exists():
        content_path = samples / "content.jpg"
        style_path = samples / "style.jpg"
        if content_path.exists() and style_path.exists():
            content = load_image(content_path, preprocess, device)
            style = load_image(style_path, preprocess, device)
            print("\ncontent features:")
            report_shapes(extractor(content))
            print("\nstyle features:")
            report_shapes(extractor(style))
            return

    # Fall back to noise so the script is still runnable without sample images.
    print("\nno sample images found, running on random noise")
    dummy = torch.randn(1, 3, 512, 512, device=device)
    report_shapes(extractor(dummy))


if __name__ == "__main__":
    main()
