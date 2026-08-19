import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import exposure

from infer_t3_pretrained_min import IMAGE_MEAN, IMAGE_STD, INPUT_SIZE, M, predict

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_THRESHOLD = 0.86


def images(root):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not any(token in path.name.lower() for token in ("_label_bin", "_mask", "_overlay", "_pred", "_legend"))
    )


def tensor(path, mean, std):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    size = image.shape[:2]
    enhanced = np.empty_like(image)
    for channel in range(3):
        values = image[..., channel]
        low, high = np.percentile(values, (1.0, 99.0))
        values = np.clip(values, low, high)
        values = (values - values.min()) / (values.max() - values.min() + 1e-6)
        enhanced[..., channel] = exposure.equalize_adapthist(values, clip_limit=0.02)
    x = torch.from_numpy(enhanced.transpose(2, 0, 1)).unsqueeze(0).cuda()
    return (F.interpolate(x, INPUT_SIZE, mode="bilinear") - mean) / std, size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    model = M().cuda()
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    mean = torch.tensor(IMAGE_MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, device="cuda").view(1, 3, 1, 1)
    paths = images(args.input)
    if not paths:
        raise SystemExit("No images")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for path in paths:
        x, size = tensor(path, mean, std)
        mask = F.interpolate(
            (predict(model, x) > MASK_THRESHOLD).to(torch.uint8), size, mode="nearest"
        )[0, 0].cpu().numpy() * 255
        relative = path.parent.relative_to(args.input) / f"{path.stem}_label_bin.png"
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(target)
        cases.append({"case_id": path.stem, "segmentation": relative.as_posix()})
    (args.output / "task3_predictions.json").write_text(
        json.dumps({"cases": cases}, separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
