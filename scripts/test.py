import argparse
import json
import os

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LucasDataset, test_transform
from model import get_model

CLASS_DESCRIPTIONS = {
    0: "Arable land",
    1: "Permanent crops",
    2: "Grassland",
    3: "Wooded areas",
    4: "Shrubs",
    5: "Bare surface, low or rare vegetation",
    6: "Artificial constructions and sealed areas",
    7: "Inland waters / Transitional waters and Coastal waters",
}


def _build_dataset_kwargs(args, transform=None):
    kwargs = {
        "patch_meters": args.patch_meters,
        "crop_mode": args.crop_mode,
        "filter_points": getattr(args, "filter_points", False),
        "imagery_type": getattr(args, "imagery_type", "auto"),
    }
    if getattr(args, "vhr_min_values", None) is not None:
        kwargs["vhr_min_values"] = args.vhr_min_values
    if getattr(args, "vhr_max_values", None) is not None:
        kwargs["vhr_max_values"] = args.vhr_max_values
    if transform is not None:
        kwargs["transform"] = transform
    return kwargs


def _loader_worker_kwargs(num_workers):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 1
    return kwargs


def load_experiment_config(experiment_dir):
    config_path = os.path.join(experiment_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)

    legacy_run_path = os.path.join(experiment_dir, "run.txt")
    legacy_preproc_path = os.path.join(experiment_dir, "preprocessing_config.json")
    if not os.path.exists(legacy_run_path):
        raise FileNotFoundError(
            f"Could not find config.json in {experiment_dir}, and no legacy run.txt fallback exists."
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


def resolve_test_preprocessing(args):
    config = load_experiment_config(args.experiment_dir)
    config_data = config.get("data", {})
    config_preproc = config.get("preprocessing", {})
    config_training = config.get("training", {})

    if getattr(args, "imagery_type", "auto") == "auto":
        imagery_type = config_data.get("imagery_type", "auto")
    else:
        imagery_type = getattr(args, "imagery_type", "auto")

    vhr_min_values = config_preproc.get("vhr_min_values")
    vhr_max_values = config_preproc.get("vhr_max_values")

    args.image_dir = args.image_dir or config_data.get("image_dir")
    args.csv_path = args.csv_path or config_data.get("csv_path")
    args.patch_meters = int(config_data.get("patch_meters", args.patch_meters))
    args.crop_mode = config_data.get("crop_mode", args.crop_mode)
    args.filter_points = bool(config_data.get("filter_points", getattr(args, "filter_points", False)))
    args.lr = float(config_training.get("learning_rate", args.lr))
    args.warmup_epochs = int(config_training.get("warmup_epochs", args.warmup_epochs))

    if imagery_type == "vhr" and ((vhr_min_values is None) != (vhr_max_values is None)):
        raise ValueError("VHR preprocessing in config is incomplete: expected both min and max values.")

    if imagery_type == "vhr" and vhr_min_values is None:
        raise ValueError(
            "VHR testing requires saved min-max bounds in config.json."
        )

    return imagery_type, vhr_min_values, vhr_max_values


def load_test_split(experiment_dir):
    test_ids_path = os.path.join(experiment_dir, "test_image_ids.txt")

    if os.path.exists(test_ids_path):
        print(f"Loading test image IDs from: {test_ids_path}")
        with open(test_ids_path, "r") as f:
            image_ids = [line.strip() for line in f if line.strip()]
        print(f"Found {len(image_ids)} test samples")
        return image_ids

    print(f"Test split files not found in: {experiment_dir}")
    print("Available files in experiment directory:")
    if os.path.exists(experiment_dir):
        for file in os.listdir(experiment_dir):
            print(f"  - {file}")
    return None


def load_checkpoint(model, checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    return model


def get_test_indices(dataset, test_entries):
    id_to_idx = {}
    id_counts = {}

    for idx, (image_path, lucas_id) in enumerate(dataset.image_files):
        id_to_idx[lucas_id] = idx
        id_counts[lucas_id] = id_counts.get(lucas_id, 0) + 1

    test_indices = []
    missing_ids = []
    ambiguous_ids = []

    for image_id in test_entries:
        image_id = str(image_id).strip()
        if image_id in id_to_idx:
            if id_counts.get(image_id, 0) > 1:
                ambiguous_ids.append(image_id)
            test_indices.append(id_to_idx[image_id])
        else:
            missing_ids.append(image_id)

    return test_indices, missing_ids, ambiguous_ids


def build_classification_report(labels, predictions, unique_classes):
    class_names = [CLASS_DESCRIPTIONS.get(i, f"Class {i}") for i in unique_classes]
    return classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )


def print_classification_report(labels, predictions, unique_classes):
    report = build_classification_report(labels, predictions, unique_classes)
    print("\n=== Classification Report ===")
    print(report)
    return report


def run_testing(args):
    num_workers = getattr(args, "num_workers", 2)
    imagery_type, vhr_min_values, vhr_max_values = resolve_test_preprocessing(args)
    args.imagery_type = imagery_type
    args.vhr_min_values = vhr_min_values
    args.vhr_max_values = vhr_max_values

    dataset = LucasDataset(
        args.image_dir,
        args.csv_path,
        **dict(_build_dataset_kwargs(args, transform=test_transform()), verbose=False),
    )
    print(f"Loaded full dataset with {len(dataset)} samples")
    
    split_entries = load_test_split(args.experiment_dir)
    
    if split_entries is None:
        print("ERROR: Could not find test image IDs!")
        print("Make sure you provide the correct --experiment_dir path where training logs are saved.")
        print("This should contain 'test_image_ids.txt'.")
        return

    test_indices, missing_refs, ambiguous_ids = get_test_indices(dataset, split_entries)
    if missing_refs:
        print(f"Warning: {len(missing_refs)} test split entries were not found in the current dataset")
        print(f"Missing entries: {missing_refs[:5]}..." if len(missing_refs) > 5 else f"Missing entries: {missing_refs}")
    if ambiguous_ids:
        print(f"Warning: {len(set(ambiguous_ids))} image IDs matched multiple files. The saved manifest is safer than ID-only splits.")
    print(f"Found {len(test_indices)} test samples out of {len(split_entries)} requested")

    test_ds = torch.utils.data.Subset(dataset, test_indices)
    print(f"Created test subset with {len(test_ds)} samples")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **_loader_worker_kwargs(num_workers),
    )
    num_classes = dataset.num_classes
    print(f"Number of classes: {num_classes}")
    model = get_model(num_classes, lr=args.lr, save_dir=None, warmup_epochs=args.warmup_epochs)
    load_checkpoint(model, args.checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"\n=== Running inference on {len(test_ds)} samples ===")
    pred_df = run_inference(model, test_loader, dataset, test_indices)

    print("\nCalculating per-class metrics...")
    calculate_per_class_metrics(pred_df, output_dir=args.experiment_dir)

    if args.save_predictions:
        print("\nSaving detailed predictions...")
        save_detailed_predictions_from_df(pred_df, args.experiment_dir)
    else:
        print("\nSkipping detailed predictions (use --save_predictions to enable)")


def run_inference(model, test_loader, dataset, test_indices):
    model.eval()
    device = next(model.parameters()).device
    predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Running inference")):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(len(labels)):
                sample_idx_in_test = batch_idx * test_loader.batch_size + i
                global_idx = test_indices[sample_idx_in_test]
                _image_path, lucas_id = dataset.image_files[global_idx]

                prob_row = probs[i].cpu().numpy()
                prob_dict = {f"p{j}": float(prob_row[j]) for j in range(len(prob_row))}
                max_prob = float(prob_row.max())

                predictions.append({
                    'image_id': lucas_id,
                    'true_label': int(labels[i].cpu().item()),
                    'predicted_label': int(preds[i].cpu().item()),
                    'confidence': max_prob,
                    'global_index': int(global_idx),
                } | prob_dict)

    pred_df = pd.DataFrame(predictions)
    return pred_df


def calculate_per_class_metrics(pred_df, output_dir=None):
    if pred_df is None or len(pred_df) == 0:
        print("No predictions available to calculate metrics.")
        return

    all_predictions = pred_df['predicted_label'].tolist()
    all_labels = pred_df['true_label'].tolist()

    correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
    total = len(all_labels)
    overall_accuracy = correct / total if total > 0 else 0.0
    summary_lines = []

    print(f"\n=== Test Results Summary ===")
    print(f"Overall Test Accuracy: {correct}/{total} = {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    summary_lines.append("=== Test Results Summary ===")
    summary_lines.append(f"Overall Test Accuracy: {correct}/{total} = {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

    unique_classes = sorted(pred_df['true_label'].unique())
    print(f"\n=== Per-Class Test Accuracy ===")
    summary_lines.append("")
    summary_lines.append("=== Per-Class Test Accuracy ===")
    for class_idx in unique_classes:
        class_samples = pred_df[pred_df['true_label'] == class_idx]
        if len(class_samples) > 0:
            class_accuracy = (class_samples['true_label'] == class_samples['predicted_label']).mean()
            class_name = CLASS_DESCRIPTIONS.get(class_idx, f"Class {class_idx}")
            total_samples = len(class_samples)
            correct_samples = (class_samples['true_label'] == class_samples['predicted_label']).sum()
            line = f"Class {class_idx} ({class_name}): {correct_samples}/{total_samples} = {class_accuracy:.4f} ({class_accuracy*100:.2f}%)"
            print(line)
            summary_lines.append(line)

    report = print_classification_report(all_labels, all_predictions, unique_classes)

    if output_dir:
        report_path = os.path.join(output_dir, "test_classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)

        summary_path = os.path.join(output_dir, "test_metrics.txt")
        with open(summary_path, "w") as f:
            f.write("\n".join(summary_lines))
            f.write("\n\n")
            f.write(report)

        class_names = [CLASS_DESCRIPTIONS.get(i, f"Class {i}") for i in unique_classes]
        cm = confusion_matrix(all_labels, all_predictions, labels=unique_classes)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_path = os.path.join(output_dir, "test_confusion_matrix.csv")
        cm_df.to_csv(cm_path)

        print(f"Saved test classification report to: {report_path}")
        print(f"Saved test metrics summary to: {summary_path}")
        print(f"Saved test confusion matrix to: {cm_path}")


def save_detailed_predictions_from_df(pred_df, experiment_dir):
    if pred_df is None or len(pred_df) == 0:
        print("No predictions available to save.")
        return pred_df
    pred_csv_path = os.path.join(experiment_dir, "test_predictions.csv")

    prob_cols = [c for c in pred_df.columns if c.startswith("p") and c[1:].isdigit()]
    prob_cols = sorted(prob_cols, key=lambda x: int(x[1:]))

    out_rows = []
    for _, row in pred_df.iterrows():
        image_id = row.get('image_id')
        true_label = int(row.get('true_label')) if not pd.isna(row.get('true_label')) else None
        global_index = int(row.get('global_index')) if 'global_index' in row and not pd.isna(row.get('global_index')) else None

        if prob_cols:
            probs = row[prob_cols].astype(float).values
            idxs = probs.argsort()[::-1]
            top1 = int(idxs[0])
            top1_prob = float(probs[idxs[0]])
            if len(probs) > 1:
                top2 = int(idxs[1])
                top2_prob = float(probs[idxs[1]])
            else:
                top2 = None
                top2_prob = None
        else:
            top1 = int(row.get('predicted_label')) if 'predicted_label' in row and not pd.isna(row.get('predicted_label')) else None
            top1_prob = float(row.get('confidence')) if 'confidence' in row and not pd.isna(row.get('confidence')) else None
            top2 = None
            top2_prob = None

        out_rows.append({
            'image_id': image_id,
            'true_label': true_label,
            'top1_label': top1,
            'top1_prob': top1_prob,
            'top2_label': top2,
            'top2_prob': top2_prob,
            'global_index': global_index
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(pred_csv_path, index=False)

    accuracy = (out_df['true_label'] == out_df['top1_label']).mean()
    print(f"\nDetailed Results (top-2 saved):")
    print(f"Overall Test Accuracy (top1 vs true): {accuracy:.4f} ({accuracy*100:.2f}%)")

    print(f"\n=== Per-Class Test Accuracy (using top1) ===")
    for class_idx in sorted(out_df['true_label'].dropna().unique()):
        class_mask = out_df['true_label'] == class_idx
        class_samples = out_df[class_mask]
        if len(class_samples) > 0:
            class_accuracy = (class_samples['true_label'] == class_samples['top1_label']).mean()
            class_name = CLASS_DESCRIPTIONS.get(class_idx, f"Class {class_idx}")
            total_samples = len(class_samples)
            correct_samples = (class_samples['true_label'] == class_samples['top1_label']).sum()
            print(f"Class {class_idx} ({class_name}): {correct_samples}/{total_samples} = {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")

    labels = out_df['true_label'].dropna().astype(int).tolist()
    preds = out_df['top1_label'].dropna().astype(int).tolist()
    if labels and preds:
        unique_classes = sorted(set(labels))
        print_classification_report(labels, preds, unique_classes)

    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained model on test set")
    parser.add_argument("--experiment_dir", type=str, required=True, 
                        help="Path to experiment directory with saved indices and checkpoints")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint (.ckpt file)")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to image directory")  
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to labels CSV file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers (set 0 on low-RAM systems)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (for model initialization)")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Warmup epochs used when the model was trained")
    parser.add_argument("--patch_meters", type=int, default=384,
                        help="Center patch size in meters")
    parser.add_argument("--imagery_type", choices=["auto", "orthophoto", "vhr"], default="auto",
                        help="Image-source preprocessing preset. Use 'orthophoto' or 'vhr' for explicit control.")
    parser.add_argument("--filter_points", action="store_true",
                        help="Use the interpreter-agreement filtered dataset")
    parser.add_argument("--crop_mode", type=str, default="center_crop", choices=["center_crop", "none"],
                        help="'center_crop' for patch classification, 'none' for full image classification")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save detailed predictions to CSV (takes extra time)")
    
    args = parser.parse_args()
    
    run_testing(args)
