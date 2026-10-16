from pathlib import Path
from datetime import datetime
from random import Random
from shutil import copy2
import cv2

ROOT = Path(__file__).resolve().parent
K = 3


def distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def kmeans(data):
    centers = Random(42).sample(data, K)
    labels = []
    for _ in range(100):
        new = [min(range(K), key=lambda i: distance(x, centers[i])) for x in data]
        if new == labels:
            break
        labels = new
        for i in range(K):
            group = [x for x, label in zip(data, labels) if label == i]
            if group:
                centers[i] = [sum(c) / len(group) for c in zip(*group)]
    return labels


def main():
    files, data = [], []
    for path in sorted((ROOT / "ML_Data").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            pixels = image.tolist()
            h, w = len(pixels), len(pixels[0])
            data.append([pixels[y * h // 16][x * w // 16] for y in range(16) for x in range(16)])
            files.append(path)
    if len(data) < K:
        raise SystemExit(f"Need at least {K} readable images; found {len(data)}.")
    output = ROOT / f"Face_Groups_{datetime.now():%Y%m%d_%H%M%S_%f}"
    for path, label in zip(files, kmeans(data)):
        target = output / f"group_{label + 1}" / path.relative_to(ROOT / "ML_Data")
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(path, target)
        print(f"{path.name} -> group {label + 1}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
