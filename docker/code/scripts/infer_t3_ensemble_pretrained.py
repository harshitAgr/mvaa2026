import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "baseline" / "task3"), str(ROOT / "scripts")]

from dataset import IMAGENET_MEAN, IMAGENET_STD
from infer_t3_d4tta import load_ckpt_config, load_state_dict, predict_probs_d4
from infer_t3_pretrained_min import IMAGE_MEAN, IMAGE_STD, M, predict
from model_factory import get_model

SIZE = (448, 800)
SUFFIXES = {".png", ".jpg"}
WEIGHTS = (0.60, 0.30, 0.10)
THRESHOLD = 0.45
PRETRAINED_THRESHOLD = 0.70
CALIBRATION_SHIFT = math.log(THRESHOLD / (1 - THRESHOLD)) - math.log(
    PRETRAINED_THRESHOLD / (1 - PRETRAINED_THRESHOLD)
)


def load_member(path):
    checkpoint, config = load_ckpt_config(path)
    model = get_model(
        arch=str(config.get("arch", "unetplusplus")),
        encoder_name=str(config.get("encoder_name", "efficientnet-b4")),
        encoder_weights=config.get("encoder_weights"),
        in_channels=3,
        classes=1,
    ).cuda()
    load_state_dict(model, checkpoint)
    model.eval()
    if tuple(config.get("image_size", SIZE)) != SIZE:
        raise RuntimeError(f"unexpected input size in {path}")
    return model


def paths(root):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUFFIXES
        and not any(x in path.name.lower() for x in ("_label_bin", "_mask", "_pred"))
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ckpt-a", type=Path, required=True)
    parser.add_argument("--ckpt-b", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    a = load_member(args.ckpt_a)
    b = load_member(args.ckpt_b)
    p = M().cuda().half()
    p.load_state_dict(
        torch.load(args.pretrained_weights, map_location="cpu", weights_only=True), strict=True
    )
    p.eval()

    mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)
    p_mean = torch.tensor(IMAGE_MEAN, device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
    p_std = torch.tensor(IMAGE_STD, device="cuda", dtype=torch.float16).view(1, 3, 1, 1)

    images = paths(args.input)
    if not images:
        raise SystemExit("No images")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for path in images:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        native_size = image.shape[:2]
        x = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).cuda()
        x = F.interpolate(x, SIZE, mode="bilinear")
        ab = (x - mean) / std
        pp = predict(p, (x.half() - p_mean) / p_std).float()
        pp = torch.sigmoid(torch.logit(pp, eps=1e-6) + CALIBRATION_SHIFT)
        probability = (
            WEIGHTS[0] * predict_probs_d4(a, ab, use_amp=True)
            + WEIGHTS[1] * predict_probs_d4(b, ab, use_amp=True)
            + WEIGHTS[2] * pp
        )
        mask = F.interpolate(
            (probability > THRESHOLD).to(torch.uint8), native_size, mode="nearest"
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
