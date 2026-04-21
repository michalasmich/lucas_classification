import argparse
import warnings
from pathlib import Path

import cv2
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.errors import NotGeoreferencedWarning

from dataset import IMAGENET_MEAN, IMAGENET_STD, LucasDataset, val_transform


matplotlib.use("Agg")


def resolve_default_csv_path():
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "csv" / "LUCAS-Master_2025_v6.csv",
        repo_root / "LUCAS-Master_2025_v6.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def denormalize_tensor(tensor):
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


def stretch_for_display(image, lower_pct=2.0, upper_pct=98.0):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] > 3:
        image = image[:, :, :3]

    image = image.astype(np.float32)
    stretched = np.zeros_like(image, dtype=np.float32)

    for band_idx in range(image.shape[2]):
        band = image[:, :, band_idx]
        finite_mask = np.isfinite(band)
        if not finite_mask.any():
            continue
        band_values = band[finite_mask]
        low = np.percentile(band_values, lower_pct)
        high = np.percentile(band_values, upper_pct)
        if high <= low:
            stretched[:, :, band_idx] = np.clip(band, 0.0, 1.0)
        else:
            stretched[:, :, band_idx] = np.clip((band - low) / (high - low), 0.0, 1.0)

    return stretched


def select_sample(dataset, requested_image_id=None):
    if not dataset.image_files:
        raise RuntimeError(f"No dataset images found in {dataset.image_dir}")

    if requested_image_id is None:
        return dataset.image_files[0]

    requested_image_id = str(requested_image_id)
    for path, image_id in dataset.image_files:
        if image_id == requested_image_id:
            return path, image_id

    raise ValueError(f"Image ID {requested_image_id} was not found in {dataset.image_dir}")


def load_full_image_and_crop(dataset, image_path, image_id):
    image_path = str(image_path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        try:
            with rasterio.open(image_path) as src:
                width, height = src.width, src.height
                center_x, center_y = dataset._resolve_crop_center(src, image_id)

                if dataset.crop_mode == "none":
                    x0, x1, y0, y1 = 0, width, 0, height
                else:
                    x0, x1, y0, y1 = dataset._get_crop_bounds(
                        width,
                        height,
                        center_x,
                        center_y,
                        image_id,
                        src.transform,
                    )

                full_image = dataset._read_rgb_bands(src, image_path)
                crop_image = dataset._read_rgb_bands(src, image_path, window=((y0, y1), (x0, x1)))

                return {
                    "reader": "rasterio",
                    "driver": src.driver,
                    "band_count": src.count,
                    "dtype": src.dtypes[0],
                    "width": width,
                    "height": height,
                    "resolution": dataset._get_resolution(image_id, src.transform),
                    "crs": str(src.crs),
                    "transform": str(src.transform),
                    "full_image": full_image,
                    "crop_image": crop_image,
                    "crop_bounds": (x0, y0, x1, y1),
                    "center_xy": (center_x, center_y),
                }
        except Exception as exc:
            img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                raise RuntimeError(f"Could not read image {image_path}: {exc}") from exc

            if img_bgr.ndim == 2:
                full_image = np.stack([img_bgr, img_bgr, img_bgr], axis=-1)
            elif img_bgr.shape[2] >= 3:
                full_image = cv2.cvtColor(img_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
            else:
                full_image = np.repeat(img_bgr[:, :, :1], 3, axis=2)

            height, width = full_image.shape[:2]
            center_x, center_y = width // 2, height // 2
            if dataset.crop_mode == "none":
                x0, x1, y0, y1 = 0, width, 0, height
            else:
                x0, x1, y0, y1 = dataset._get_crop_bounds(
                    width,
                    height,
                    center_x,
                    center_y,
                    image_id,
                    rasterio.transform.Affine.identity(),
                )

            crop_image = full_image[y0:y1, x0:x1, :3]
            return {
                "reader": "opencv",
                "driver": Path(image_path).suffix.lower(),
                "band_count": full_image.shape[2],
                "dtype": str(full_image.dtype),
                "width": width,
                "height": height,
                "resolution": dataset._get_resolution(image_id, rasterio.transform.Affine.identity()),
                "crs": "None",
                "transform": "Affine.identity()",
                "full_image": full_image,
                "crop_image": crop_image,
                "crop_bounds": (x0, y0, x1, y1),
                "center_xy": (center_x, center_y),
            }


def inspect_sample(dataset, image_path, image_id, source_label):
    raw_info = load_full_image_and_crop(dataset, image_path, image_id)

    try:
        current_crop = dataset._read_image_with_rasterio(image_path, image_id)
    except Exception:
        current_crop = dataset._read_image_with_opencv(image_path, image_id)

    model_tensor = dataset.transform(current_crop)
    model_input_display = denormalize_tensor(model_tensor.detach().cpu()).permute(1, 2, 0).numpy()
    model_input_display = np.clip(model_input_display, 0.0, 1.0)

    reverse_mapping = {mapped: original for original, mapped in dataset.label_mapping.items()}
    mapped_label = dataset.id_to_label[image_id]
    original_label = reverse_mapping[mapped_label]

    current_crop_array = np.asarray(current_crop)
    scaled_crop_display = np.clip(current_crop_array, 0.0, 1.0)
    return {
        "source_label": source_label,
        "image_id": image_id,
        "image_path": str(image_path),
        "mapped_label": mapped_label,
        "original_label": original_label,
        "raw_info": raw_info,
        "raw_display": stretch_for_display(raw_info["full_image"]),
        "crop_display_human": stretch_for_display(raw_info["crop_image"]),
        "scaled_crop_display": scaled_crop_display,
        "model_input_display": model_input_display,
        "current_crop_min": float(current_crop_array.min()),
        "current_crop_max": float(current_crop_array.max()),
        "current_crop_mean": float(current_crop_array.mean()),
        "current_crop_std": float(current_crop_array.std()),
        "current_crop_zero_fraction": float(np.mean(current_crop_array <= 1e-6)),
        "current_crop_one_fraction": float(np.mean(current_crop_array >= 1.0 - 1e-6)),
        "tensor_min": float(model_tensor.min().item()),
        "tensor_max": float(model_tensor.max().item()),
        "tensor_mean": float(model_tensor.mean().item()),
        "tensor_shape": tuple(model_tensor.shape),
        "preprocessing_summary": dataset.preprocessing_summary,
        "effective_scaling": dataset.describe_effective_scaling(raw_info["dtype"]),
    }


def plot_inspection(samples, output_path):
    fig, axes = plt.subplots(len(samples), 5, figsize=(22, 6 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, sample in enumerate(samples):
        ax_full, ax_crop_human, ax_crop_scaled, ax_model, ax_text = axes[row_idx]
        raw_info = sample["raw_info"]
        x0, y0, x1, y1 = raw_info["crop_bounds"]
        center_x, center_y = raw_info["center_xy"]

        ax_full.imshow(sample["raw_display"])
        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
        )
        ax_full.add_patch(rect)
        ax_full.scatter([center_x], [center_y], c="red", s=20)
        ax_full.set_title(f"{sample['source_label']}: full image with crop")
        ax_full.axis("off")

        ax_crop_human.imshow(sample["crop_display_human"])
        ax_crop_human.set_title(
            f"{sample['source_label']}: cropped patch for human display\n"
            f"{raw_info['crop_bounds'][2] - raw_info['crop_bounds'][0]} x "
            f"{raw_info['crop_bounds'][3] - raw_info['crop_bounds'][1]} px"
        )
        ax_crop_human.axis("off")

        ax_crop_scaled.imshow(sample["scaled_crop_display"])
        ax_crop_scaled.set_title(
            f"{sample['source_label']}: exact scaled crop\n"
            f"before resize / before Normalize"
        )
        ax_crop_scaled.axis("off")

        ax_model.imshow(sample["model_input_display"])
        ax_model.set_title(
            f"{sample['source_label']}: exact tensor image\n"
            f"before Normalize {sample['tensor_shape'][1]} x {sample['tensor_shape'][2]}"
        )
        ax_model.axis("off")

        scaling_warning = ""
        if sample["current_crop_max"] > 1.0:
            scaling_warning = "\nWARNING: crop values exceed 1.0 before Normalize"
        dark_warning = ""
        if sample["current_crop_mean"] < 0.10:
            dark_warning = "\nNOTE: scaled crop mean is low; tensor input will look dark"
        clipping_warning = ""
        if sample["current_crop_one_fraction"] > 0.01:
            clipping_warning = "\nNOTE: more than 1% of pixels are clipped at 1.0"

        meta_text = (
            f"image_id: {sample['image_id']}\n"
            f"image_path: {sample['image_path']}\n"
            f"STR25: {sample['original_label']} | mapped label: {sample['mapped_label']}\n"
            f"imagery_type: {sample['preprocessing_summary'].get('imagery_type')}\n"
            f"scaling: {sample['preprocessing_summary'].get('scaling')}\n"
            f"effective scaling: {sample['effective_scaling']}\n"
            f"reader: {raw_info['reader']} | driver: {raw_info['driver']}\n"
            f"bands: {raw_info['band_count']} | dtype: {raw_info['dtype']}\n"
            f"size: {raw_info['width']} x {raw_info['height']}\n"
            f"resolution: {raw_info['resolution']:.3f} m/px\n"
            f"crop_mode: center_crop\n"
            f"crop_bounds: x[{x0}:{x1}] y[{y0}:{y1}]\n"
            f"current crop min/max/mean: {sample['current_crop_min']:.4f} / "
            f"{sample['current_crop_max']:.4f} / {sample['current_crop_mean']:.4f}\n"
            f"current crop std: {sample['current_crop_std']:.4f}\n"
            f"fraction at 0.0: {sample['current_crop_zero_fraction']:.4%}\n"
            f"fraction at 1.0: {sample['current_crop_one_fraction']:.4%}\n"
            f"tensor min/max/mean: {sample['tensor_min']:.4f} / "
            f"{sample['tensor_max']:.4f} / {sample['tensor_mean']:.4f}"
            f"{scaling_warning}{dark_warning}{clipping_warning}"
        )
        if sample["preprocessing_summary"].get("vhr_min_values") is not None:
            meta_text += (
                f"\nvhr mins: {sample['preprocessing_summary']['vhr_min_values']}\n"
                f"vhr maxs: {sample['preprocessing_summary']['vhr_max_values']}"
            )
        ax_text.text(
            0.0,
            1.0,
            meta_text,
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )
        ax_text.set_title(f"{sample['source_label']}: metadata")
        ax_text.axis("off")

    fig.suptitle("LUCAS preprocessing inspection", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_dataset(image_dir, csv_path, patch_meters, input_size, imagery_type, vhr_min_values=None, vhr_max_values=None):
    return LucasDataset(
        image_dir=image_dir,
        label_csv=csv_path,
        patch_meters=patch_meters,
        transform=val_transform((input_size, input_size)),
        crop_mode="center_crop",
        imagery_type=imagery_type,
        vhr_min_values=vhr_min_values,
        vhr_max_values=vhr_max_values,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect how the preprocessing pipeline feeds orthophoto and VHR images to the model.")
    parser.add_argument("--csv_path", default=str(resolve_default_csv_path()), help="Path to the LUCAS CSV")
    parser.add_argument("--ortho_dir", required=True, help="Directory with orthophoto images")
    parser.add_argument("--vhr_dir", required=True, help="Directory with VHR images")
    parser.add_argument("--ortho_image_id", help="Optional orthophoto image ID to inspect")
    parser.add_argument("--vhr_image_id", help="Optional VHR image ID to inspect")
    parser.add_argument("--patch_meters", type=int, default=384, help="Patch size in meters")
    parser.add_argument("--input_size", type=int, default=224, help="Square model input size after resize")
    parser.add_argument("--vhr_min_values", type=float, nargs=3, metavar=("R_MIN", "G_MIN", "B_MIN"),
                        help="Optional per-band mins for VHR min-max scaling")
    parser.add_argument("--vhr_max_values", type=float, nargs=3, metavar=("R_MAX", "G_MAX", "B_MAX"),
                        help="Optional per-band maxs for VHR min-max scaling")
    parser.add_argument(
        "--output_path",
        default=str(Path(__file__).resolve().parent / "plots" / "preprocessing_inspection.png"),
        help="Where to save the inspection figure",
    )
    args = parser.parse_args()

    ortho_dataset = build_dataset(args.ortho_dir, args.csv_path, args.patch_meters, args.input_size, imagery_type="orthophoto")
    vhr_dataset = build_dataset(
        args.vhr_dir,
        args.csv_path,
        args.patch_meters,
        args.input_size,
        imagery_type="vhr",
        vhr_min_values=args.vhr_min_values,
        vhr_max_values=args.vhr_max_values,
    )

    ortho_path, ortho_id = select_sample(ortho_dataset, args.ortho_image_id)
    vhr_path, vhr_id = select_sample(vhr_dataset, args.vhr_image_id)

    samples = [
        inspect_sample(ortho_dataset, ortho_path, ortho_id, "Orthophoto"),
        inspect_sample(vhr_dataset, vhr_path, vhr_id, "VHR"),
    ]
    output_path = Path(args.output_path)
    plot_inspection(samples, output_path)
    print(f"Saved preprocessing inspection figure to: {output_path}")
    print(f"Orthophoto sample: {ortho_id} -> {ortho_path}")
    print(f"VHR sample: {vhr_id} -> {vhr_path}")
