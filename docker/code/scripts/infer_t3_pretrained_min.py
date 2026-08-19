import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

MODEL_NAME = "vit_base_patch16_dinov3"
INPUT_SIZE = (448, 800)
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".png", ".jpg"}
MASK_THRESHOLD = 0.7
TTA_FLIPS = ((3,), (2,), (2, 3))


class C(nn.Module):
    def __init__(self, a, b, k=3, p=1):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(a, b, k, padding=p, bias=False),
            nn.BatchNorm2d(b),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.m(x)


class S(nn.Module):
    def __init__(self, c):
        super().__init__()
        n = max(1, c // 16)
        self.m = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, n, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(n, c, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.m(x)


class P(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.a = nn.ModuleList([nn.Conv2d(c, 256, 1, bias=False) for c in inputs])
        self.b = nn.ModuleList([nn.Conv2d(c, 256, 1, bias=False) for c in inputs])
        self.c = nn.ModuleList([nn.Conv2d(256, 512, 1) for _ in inputs])
        self.d = nn.ModuleList(
            [nn.Sequential(C(256, c, 1, 0), C(c, c), S(c)) for c in outputs]
        )
        self.e = nn.ModuleList([nn.Conv2d(256, c, 1, bias=False) for c in outputs])

    def forward(self, features):
        outputs = []
        for i, feature in enumerate(features):
            shared = self.a[i](feature)
            specific = self.b[i](feature)
            gamma, beta = torch.chunk(self.c[i](shared), 2, 1)
            value = specific * torch.sigmoid(gamma) + beta
            outputs.append(self.d[i](value) + self.e[i](value))
        return outputs


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            features_only=True,
            out_indices=(2, 5, 8, 11),
        )
        self.b = P(self.a.feature_info.channels(), (96, 128, 192, 256))
        self.c = C(3, 64)
        self.d = nn.Sequential(nn.MaxPool2d(2), C(64, 96))
        self.e = C(448, 192)
        self.f = C(320, 128)
        self.g = C(320, 96)
        self.h = C(160, 96)
        self.i = nn.Sequential(C(96, 48), nn.Conv2d(48, 1, 1))
        self.j = nn.Sequential(C(144, 96), nn.Dropout2d(0.1), nn.Conv2d(96, 1, 1))

    def forward(self, image):
        size = image.shape[-2:]
        s2 = self.c(image)
        s4 = self.d(s2)
        f1, f2, f3, f4 = self.b(self.a(image))
        x = self.e(torch.cat((F.interpolate(f4, f3.shape[-2:], mode="bilinear"), f3), 1))
        x = self.f(torch.cat((F.interpolate(x, f2.shape[-2:], mode="bilinear"), f2), 1))
        x = self.g(
            torch.cat(
                (F.interpolate(x, f1.shape[-2:], mode="bilinear"), f1, F.interpolate(s4, f1.shape[-2:], mode="bilinear")),
                1,
            )
        )
        x = self.h(torch.cat((F.interpolate(x, s2.shape[-2:], mode="bilinear"), s2), 1))
        x = F.interpolate(x, size, mode="bilinear")
        y = torch.sigmoid(self.i(x)).detach() * x[:, :48]
        return self.j(torch.cat((x, y), 1))


def images(root):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def tensor(path, mean, std):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    size = image.shape[:2]
    x = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).cuda().half()
    return (F.interpolate(x, INPUT_SIZE, mode="bilinear") - mean) / std, size


@torch.inference_mode()
def predict(model, x):
    result = torch.sigmoid(model(x))
    for dims in TTA_FLIPS:
        result += torch.flip(torch.sigmoid(model(torch.flip(x, dims))), dims)
    return result / 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    model = M().cuda().half()
    model.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True), strict=True
    )
    model.eval()
    mean = torch.tensor(IMAGE_MEAN, device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, device="cuda", dtype=torch.float16).view(1, 3, 1, 1)
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
