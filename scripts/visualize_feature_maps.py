import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IMAGENET_MEAN, IMAGENET_STD, LucasDataset, val_transform
from model import CLASS_NAMES, get_model


matplotlib.use("Agg")


DEFAULT_INPUT_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0
DEFAULT_PATCH_METERS = 384
DEFAULT_CROP_MODE = "center_crop"
DEFAULT_FILTER_POINTS = False
DEFAULT_IMAGERY_TYPE = "auto"
DEFAULT_LR = 1e-4
DEFAULT_WARMUP_EPOCHS = 0
DEFAULT_CMAP = "rainbow"
DEFAULT_OVERLAY_ALPHA = 0.45
DEFAULT_EXPLAIN_METHOD = "hirescam"
DEFAULT_TARGET_LAYERS = ["layer3", "layer4"]
DEFAULT_CAM_TARGET = "pred"
DEFAULT_CAM_NORM_LOWER = 1.0
DEFAULT_CAM_NORM_UPPER = 99.0


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


def load_experiment_config(experiment_dir):
    config_path = os.path.join(experiment_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    legacy_run_path = os.path.join(experiment_dir, "run.txt")
    legacy_preproc_path = os.path.join(experiment_dir, "preprocessing_config.json")
    if not os.path.exists(legacy_run_path):
        raise FileNotFoundError(
            f"Missing config.json in {experiment_dir}, and no legacy run.txt fallback exists."
        )

    legacy = {}
    with open(legacy_run_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            legacy[key.strip()] = value.strip()

    preprocessing = {}
    if os.path.exists(legacy_preproc_path):
        with open(legacy_preproc_path, "r", encoding="utf-8") as f:
            preprocessing = json.load(f)

    def _as_bool(value):
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def _as_int(value, default=0):
        try:
            return int(value)
        except Exception:
            return default

    def _as_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    return {
        "data": {
            "image_dir": legacy.get("image_dir"),
            "csv_path": legacy.get("csv_path"),
            "imagery_type": legacy.get("imagery_type", "auto"),
            "patch_meters": _as_int(legacy.get("patch_meters"), 384),
            "crop_mode": legacy.get("crop_mode", "center_crop"),
            "filter_points": _as_bool(legacy.get("filter_points", "False")),
        },
        "training": {
            "learning_rate": _as_float(legacy.get("learning_rate"), 1e-4),
            "warmup_epochs": _as_int(legacy.get("warmup_epochs"), 0),
        },
        "preprocessing": preprocessing,
        "checkpoints": {
            "best_checkpoint_path": legacy.get("best_checkpoint_path"),
            "last_checkpoint_path": legacy.get("last_checkpoint_path"),
        },
    }


def apply_runtime_defaults(args):
    config = load_experiment_config(args.experiment_dir)
    data_cfg = config.get("data", {})
    preproc_cfg = config.get("preprocessing", {})
    training_cfg = config.get("training", {})

    args.image_dir = data_cfg.get("image_dir")
    if not args.image_dir:
        raise ValueError("config.json does not contain data.image_dir.")

    args.csv_path = data_cfg.get("csv_path") or str(resolve_default_csv_path())
    args.patch_meters = int(data_cfg.get("patch_meters", DEFAULT_PATCH_METERS))
    args.crop_mode = data_cfg.get("crop_mode", DEFAULT_CROP_MODE)
    if args.crop_mode not in {"center_crop", "none"}:
        args.crop_mode = DEFAULT_CROP_MODE

    args.filter_points = bool(data_cfg.get("filter_points", DEFAULT_FILTER_POINTS))
    args.imagery_type = data_cfg.get("imagery_type", DEFAULT_IMAGERY_TYPE)
    if args.imagery_type not in {"auto", "orthophoto", "vhr"}:
        args.imagery_type = DEFAULT_IMAGERY_TYPE

    args.vhr_min_values = preproc_cfg.get("vhr_min_values")
    args.vhr_max_values = preproc_cfg.get("vhr_max_values")

    args.input_size = DEFAULT_INPUT_SIZE
    args.batch_size = DEFAULT_BATCH_SIZE
    args.num_workers = DEFAULT_NUM_WORKERS
    args.lr = float(training_cfg.get("learning_rate", DEFAULT_LR))
    args.warmup_epochs = int(training_cfg.get("warmup_epochs", DEFAULT_WARMUP_EPOCHS))

    args.cmap = DEFAULT_CMAP
    args.overlay_alpha = DEFAULT_OVERLAY_ALPHA
    args.explain_method = DEFAULT_EXPLAIN_METHOD
    args.target_layers_list = list(DEFAULT_TARGET_LAYERS)
    args.cam_target = DEFAULT_CAM_TARGET
    args.cam_norm_lower = DEFAULT_CAM_NORM_LOWER
    args.cam_norm_upper = DEFAULT_CAM_NORM_UPPER


def _loader_worker_kwargs(num_workers):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 1
    return kwargs


def resolve_preprocessing(args):
    imagery_type = args.imagery_type
    vhr_min_values = args.vhr_min_values
    vhr_max_values = args.vhr_max_values

    if imagery_type == "vhr" and ((vhr_min_values is None) != (vhr_max_values is None)):
        raise ValueError("VHR preprocessing in config is incomplete: expected both min and max values.")

    if imagery_type == "vhr" and vhr_min_values is None:
        raise ValueError("VHR visualization needs saved min-max bounds in config.json.")

    return imagery_type, vhr_min_values, vhr_max_values


def build_dataset(args, imagery_type, vhr_min_values, vhr_max_values):
    kwargs = {
        "patch_meters": args.patch_meters,
        "crop_mode": args.crop_mode,
        "filter_points": args.filter_points,
        "imagery_type": imagery_type,
        "transform": val_transform((args.input_size, args.input_size)),
        "verbose": False,
    }
    if vhr_min_values is not None:
        kwargs["vhr_min_values"] = vhr_min_values
    if vhr_max_values is not None:
        kwargs["vhr_max_values"] = vhr_max_values
    return LucasDataset(args.image_dir, args.csv_path, **kwargs)


def load_split_entries(experiment_dir, split_name):
    ids_path = os.path.join(experiment_dir, f"{split_name}_image_ids.txt")

    if os.path.exists(ids_path):
        with open(ids_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    raise FileNotFoundError(
        f"Could not find split files for '{split_name}' in {experiment_dir}. "
        f"Expected {split_name}_image_ids.txt."
    )


def map_split_to_indices(dataset, split_entries):
    id_to_indices = {}

    for idx, (_image_path, image_id) in enumerate(dataset.image_files):
        id_to_indices.setdefault(image_id, []).append(idx)

    split_indices = []
    missing_entries = []
    ambiguous_entries = []

    for image_id in split_entries:
        image_id = str(image_id).strip()
        candidates = id_to_indices.get(image_id, [])
        if not candidates:
            missing_entries.append(image_id)
            continue
        if len(candidates) > 1:
            ambiguous_entries.append(image_id)
        split_indices.append(candidates[0])

    return split_indices, missing_entries, sorted(set(ambiguous_entries))


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model


def _class_name(class_idx):
    if 0 <= class_idx < len(CLASS_NAMES):
        return CLASS_NAMES[class_idx]
    return f"Class {class_idx}"


def _safe_name(value):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


CONFIDENCE_BUCKET_ORDER = ["high", "medium", "low"]


def denormalize_tensor_image(image_tensor):
    mean = torch.tensor(IMAGENET_MEAN, dtype=image_tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image_tensor.dtype).view(3, 1, 1)
    rgb = image_tensor.detach().cpu() * std + mean
    rgb = rgb.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return rgb.astype(np.float32)


def normalize_cam_for_display(cam_map, lower_pct=1.0, upper_pct=99.0):
    cam = np.asarray(cam_map, dtype=np.float32)
    finite = np.isfinite(cam)
    if not finite.any():
        return np.zeros_like(cam, dtype=np.float32)

    values = cam[finite]
    lo = float(np.percentile(values, lower_pct))
    hi = float(np.percentile(values, upper_pct))
    if hi <= lo:
        return np.zeros_like(cam, dtype=np.float32)
    return np.clip((cam - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def make_overlay(rgb_image, cam_map, cmap_name="rainbow", alpha=0.45):
    color_map = matplotlib.colormaps[cmap_name]
    heat_rgb = color_map(cam_map)[..., :3].astype(np.float32)
    return np.clip((1.0 - alpha) * rgb_image + alpha * heat_rgb, 0.0, 1.0)


class MultiLayerCAM:
    def __init__(self, lit_model, target_layer_names, method="hirescam"):
        self.lit_model = lit_model
        self.model = lit_model.model.network
        self.method = method
        self.target_layer_names = target_layer_names
        self.activations = {}
        self.gradients = {}
        self.handles = []
        self._register_hooks()

    def _resolve_layer(self, layer_name):
        if not hasattr(self.model, layer_name):
            raise ValueError(f"Backbone has no layer named '{layer_name}'.")
        module = getattr(self.model, layer_name)
        if isinstance(module, torch.nn.Sequential):
            return module[-1]
        return module

    def _register_hooks(self):
        for layer_name in self.target_layer_names:
            module = self._resolve_layer(layer_name)

            def forward_hook(_module, _input, output, lname=layer_name):
                self.activations[lname] = output

            def backward_hook(_module, _grad_input, grad_output, lname=layer_name):
                self.gradients[lname] = grad_output[0]

            self.handles.append(module.register_forward_hook(forward_hook))
            self.handles.append(module.register_full_backward_hook(backward_hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _single_layer_cam(self, activations, gradients):
        if self.method == "gradcam":
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            raw = (weights * activations).sum(dim=1, keepdim=True)
        else:
            raw = (gradients * activations).sum(dim=1, keepdim=True)
        return F.relu(raw)

    def generate(self, input_tensor, target_class):
        self.activations = {}
        self.gradients = {}

        self.lit_model.zero_grad(set_to_none=True)
        logits = self.lit_model(input_tensor)
        score = logits[:, int(target_class)].sum()
        score.backward(retain_graph=False)

        _, _, in_h, in_w = input_tensor.shape
        layer_cams = []
        for layer_name in self.target_layer_names:
            if layer_name not in self.activations or layer_name not in self.gradients:
                raise RuntimeError(f"Missing hooks for layer '{layer_name}' while generating CAM.")
            cam = self._single_layer_cam(self.activations[layer_name], self.gradients[layer_name])
            cam = F.interpolate(cam, size=(in_h, in_w), mode="bilinear", align_corners=False)
            layer_cams.append(cam.detach().cpu().numpy()[0, 0])

        fused = np.mean(np.stack(layer_cams, axis=0), axis=0)
        return fused


def collect_examples(args, model, loader, dataset, split_indices, num_classes):
    candidates = {class_idx: {"correct": [], "incorrect": []} for class_idx in range(num_classes)}

    seen = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Scanning full {args.split} split"):
            images = batch["image"].to(next(model.parameters()).device)
            labels = batch["label"].to(next(model.parameters()).device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            batch_size = labels.size(0)
            for idx_in_batch in range(batch_size):
                true_label = int(labels[idx_in_batch].item())
                pred_label = int(preds[idx_in_batch].item())
                confidence = float(probs[idx_in_batch, pred_label].item())

                if true_label not in candidates:
                    continue

                group = "correct" if true_label == pred_label else "incorrect"

                split_position = seen + idx_in_batch
                global_idx = split_indices[split_position]
                _image_path, image_id = dataset.image_files[global_idx]

                sample = {
                    "image_id": image_id,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "confidence": confidence,
                    "global_index": int(global_idx),
                    "split_position": int(split_position),
                }
                candidates[true_label][group].append(sample)

            seen += batch_size

    selection = _select_confidence_bucket_representatives(candidates)
    return selection


def _select_representative_from_bucket(bucket_samples):
    if not bucket_samples:
        return None
    confidences = np.asarray([sample["confidence"] for sample in bucket_samples], dtype=np.float32)
    target_conf = float(np.median(confidences))
    representative = min(bucket_samples, key=lambda sample: abs(sample["confidence"] - target_conf))
    return dict(representative)


def _select_confidence_bucket_representatives(candidates):
    selection = {}
    for class_idx, class_groups in candidates.items():
        selection[class_idx] = {}
        for group_name, samples in class_groups.items():
            bucket_map = {bucket: None for bucket in CONFIDENCE_BUCKET_ORDER}
            if samples:
                sorted_samples = sorted(samples, key=lambda sample: sample["confidence"])
                split_chunks = np.array_split(np.arange(len(sorted_samples)), 3)
                ascending_bucket_order = ["low", "medium", "high"]
                for bucket_name, chunk_indices in zip(ascending_bucket_order, split_chunks):
                    if len(chunk_indices) == 0:
                        continue
                    bucket_samples = [sorted_samples[int(idx)] for idx in chunk_indices]
                    representative = _select_representative_from_bucket(bucket_samples)
                    if representative is not None:
                        representative["confidence_bucket"] = bucket_name
                    bucket_map[bucket_name] = representative
            selection[class_idx][group_name] = bucket_map
    return selection


def attach_cam_visualizations(args, dataset, selection, cam_extractor, device):
    all_samples = []
    for class_idx in selection:
        for group_name in ("correct", "incorrect"):
            for bucket_name in CONFIDENCE_BUCKET_ORDER:
                sample = selection[class_idx][group_name].get(bucket_name)
                if sample is not None:
                    all_samples.append(sample)

    for sample in tqdm(all_samples, desc="Computing CAM overlays"):
        item = dataset[sample["global_index"]]
        input_tensor = item["image"].unsqueeze(0).to(device)

        if args.cam_target == "true":
            cam_target_class = int(sample["true_label"])
        else:
            cam_target_class = int(sample["predicted_label"])

        cam_map_raw = cam_extractor.generate(input_tensor, cam_target_class)
        cam_map = normalize_cam_for_display(
            cam_map_raw,
            lower_pct=args.cam_norm_lower,
            upper_pct=args.cam_norm_upper,
        )
        rgb = denormalize_tensor_image(input_tensor[0])
        overlay = make_overlay(rgb, cam_map, cmap_name=args.cmap, alpha=args.overlay_alpha)

        sample["cam_map"] = cam_map
        sample["overlay"] = overlay


def _split_display_name(split_name):
    if split_name == "val":
        return "Validation Set"
    if split_name == "test":
        return "Test Set"
    return str(split_name)


def save_class_figure(class_idx, class_samples, output_path, split_name, args):
    class_name = _class_name(class_idx)
    images_per_group = len(CONFIDENCE_BUCKET_ORDER)
    fig, axes = plt.subplots(
        2,
        images_per_group,
        figsize=(3.2 * images_per_group, 6.2),
        squeeze=False,
        constrained_layout=True,
    )

    row_specs = [("correct", 0, "Correct"), ("incorrect", 1, "Incorrect")]
    for group_name, row_idx, row_label in row_specs:
        for col_idx, bucket_name in enumerate(CONFIDENCE_BUCKET_ORDER):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            sample = class_samples[group_name].get(bucket_name)
            if sample is None:
                ax.text(0.5, 0.55, "N/A", ha="center", va="center", fontsize=11)
                ax.text(0.5, 0.40, bucket_name.capitalize(), ha="center", va="center", fontsize=9)
                continue

            ax.imshow(sample["overlay"])
            title = (
                f"{bucket_name.capitalize()} Confidence\n"
                f"Lucas ID: {sample['image_id']}\n"
                f"Actual: {sample['true_label']} | Predicted: {sample['predicted_label']}\n"
                f"Confidence: {sample['confidence'] * 100:.1f}%"
            )
            ax.set_title(title, fontsize=8)
        axes[row_idx, 0].set_ylabel(row_label, fontsize=11)

    split_display = _split_display_name(split_name)
    fig.suptitle(
        f"{split_display} | Class {class_idx}: {class_name}",
        fontsize=11,
    )
    scalar_mappable = plt.cm.ScalarMappable(cmap=args.cmap, norm=plt.Normalize(0.0, 1.0))
    scalar_mappable.set_array([])
    cbar = fig.colorbar(
        scalar_mappable,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.025,
        pad=0.05,
        shrink=0.35,
    )
    cbar.ax.set_title("Intensity", fontsize=8, pad=2)
    cbar.ax.set_xlabel(
        f"Visualization produced using {args.explain_method.upper()}",
        fontsize=7,
        labelpad=4,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(args):
    torch.manual_seed(42)
    np.random.seed(42)
    apply_runtime_defaults(args)
    if args.images_per_group != 3:
        print("Forcing 3 columns for confidence buckets: high | medium | low")
    args.images_per_group = 3

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.experiment_dir) / f"feature_maps_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    imagery_type, vhr_min_values, vhr_max_values = resolve_preprocessing(args)
    dataset = build_dataset(args, imagery_type, vhr_min_values, vhr_max_values)
    print(f"Loaded dataset with {len(dataset)} samples (imagery_type={dataset.imagery_type})")

    split_entries = load_split_entries(args.experiment_dir, args.split)
    split_indices, missing_entries, ambiguous_entries = map_split_to_indices(dataset, split_entries)
    print(f"Resolved {len(split_indices)} {args.split} samples")
    if missing_entries:
        print(f"Warning: {len(missing_entries)} split entries not found in current dataset")
    if ambiguous_entries:
        print(
            f"Warning: {len(ambiguous_entries)} IDs matched multiple files. "
            "Manifest with relative paths is recommended."
        )
    if not split_indices:
        raise RuntimeError(f"No {args.split} indices resolved from split files.")

    split_subset = torch.utils.data.Subset(dataset, split_indices)
    loader = DataLoader(
        split_subset,
        batch_size=args.batch_size,
        shuffle=False,
        **_loader_worker_kwargs(args.num_workers),
    )

    model = get_model(
        num_classes=dataset.num_classes,
        lr=args.lr,
        save_dir=None,
        warmup_epochs=args.warmup_epochs,
    )
    load_checkpoint(model, args.checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on {device} and set to eval mode")

    selection = collect_examples(
        args=args,
        model=model,
        loader=loader,
        dataset=dataset,
        split_indices=split_indices,
        num_classes=dataset.num_classes,
    )

    cam_extractor = MultiLayerCAM(model, args.target_layers_list, method=args.explain_method)
    try:
        attach_cam_visualizations(args, dataset, selection, cam_extractor, device)
    finally:
        cam_extractor.close()

    for class_idx in range(dataset.num_classes):
        class_name = _class_name(class_idx)
        safe_class_name = _safe_name(class_name)
        output_path = output_dir / f"class_{class_idx:02d}_{safe_class_name}.png"
        save_class_figure(
            class_idx=class_idx,
            class_samples=selection[class_idx],
            output_path=output_path,
            split_name=args.split,
            args=args,
        )
        print(f"Saved: {output_path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train-free class-activation visualization for correct/incorrect samples per class."
    )
    parser.add_argument("--checkpoint_path", required=True, help="Path to Lightning .ckpt")
    parser.add_argument("--experiment_dir", required=True, help="Training log directory with split files")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Which split to visualize")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional custom output directory")
    parser.add_argument("--images_per_group", type=int, default=3, help="Images per row group")

    parsed_args = parser.parse_args()
    main(parsed_args)
