import argparse

import cv2
import matplotlib.pyplot as plt

from kmean import K, ROOT, kmeans


def plot_comparison(files, labels, output, show=True):
    """Show all source images in one figure, with one row per K-means group."""
    groups = [[path for path, label in zip(files, labels) if label == i] for i in range(K)]
    columns = max(len(group) for group in groups)
    fig, axes = plt.subplots(
        K, columns, figsize=(columns * 2 + 1.5, K * 3 + 1), squeeze=False
    )
    fig.subplots_adjust(
        left=0.09, right=0.99, bottom=0.07, top=0.87, hspace=0.45, wspace=0.12
    )
    fig.suptitle("K-means image comparison", fontsize=22, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.93,
        f"{len(files)} images | {K} groups | Similarity from 16 x 16 grayscale samples",
        ha="center", fontsize=11, color="#555555",
    )

    for group_index, group in enumerate(groups):
        for ax in axes[group_index]:
            ax.set_axis_off()
        axes[group_index, 0].text(
            -0.18, 0.5, f"Group {group_index + 1}\n{len(group)} images",
            transform=axes[group_index, 0].transAxes, ha="right", va="center",
            fontsize=12, fontweight="bold",
        )
        for ax, path in zip(axes[group_index], group):
            image = cv2.imread(str(path))
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            identifier, separator, name = path.stem.partition("_")
            caption = f"{name}\n{identifier}" if separator else path.stem
            ax.text(
                0.5, -0.04, caption, transform=ax.transAxes,
                ha="center", va="top", fontsize=9,
            )

    fig.savefig(output, dpi=160, facecolor="white")
    print(f"Comparison saved to {output}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare K-means image groups in one figure.")
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the comparison without opening a plot window.",
    )
    args = parser.parse_args()

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

    labels = kmeans(data)
    plot_comparison(files, labels, ROOT / "comparison.png", show=not args.no_show)


if __name__ == "__main__":
    main()
