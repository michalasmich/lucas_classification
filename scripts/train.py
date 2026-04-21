import os
import json
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import LucasDataset, estimate_vhr_minmax_from_dataset, train_transform, val_transform
from model import CLASS_NAMES, get_model

DEFAULT_LOG_ROOT = Path(r"F:\mixalis_projects\lucas\lucas_logs")
SAVE_WEIGHTS_ONLY_CHECKPOINTS = True


def _save_split_ids(log_dir, dataset, train_idx, val_idx, test_idx):
    split_files = {
        "train_image_ids.txt": train_idx,
        "val_image_ids.txt": val_idx,
        "test_image_ids.txt": test_idx,
    }

    for filename, split_indices in split_files.items():
        path = os.path.join(log_dir, filename)
        with open(path, "w") as f:
            for idx in split_indices:
                f.write(f"{dataset.image_files[idx][1]}\n")


def _to_serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return str(value)


def _save_experiment_config(
    config_path,
    args,
    train_count,
    val_count,
    test_count,
    log_dir,
    num_workers,
    trainer_precision,
    num_classes,
    preprocessing_summary,
    best_checkpoint_path,
    last_checkpoint_path,
):
    cli_args = _to_serializable(vars(args))
    config = {
        "cli_args": cli_args,
        "experiment": {
            "experiment_dir": str(log_dir),
            "split_ratio": "70-15-15 (train-val-test)",
            "num_samples_train": int(train_count),
            "num_samples_val": int(val_count),
            "num_samples_test": int(test_count),
            "num_classes": int(num_classes),
            "class_names": CLASS_NAMES,
        },
        "data": {
            "image_dir": str(args.image_dir),
            "csv_path": str(args.csv_path),
            "imagery_type": str(args.imagery_type),
            "patch_meters": int(args.patch_meters),
            "crop_mode": str(args.crop_mode),
            "filter_points": bool(getattr(args, "filter_points", False)),
        },
        "training": {
            "max_epochs": int(args.max_epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "warmup_epochs": int(args.warmup_epochs),
            "num_workers": int(num_workers),
            "precision": str(trainer_precision),
            "save_weights_only_checkpoints": bool(SAVE_WEIGHTS_ONLY_CHECKPOINTS),
        },
        "preprocessing": _to_serializable(preprocessing_summary),
        "checkpoints": {
            "best_checkpoint_path": _to_serializable(best_checkpoint_path),
            "last_checkpoint_path": _to_serializable(last_checkpoint_path),
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _resolve_trainer_precision():
    if not torch.cuda.is_available():
        return "32-true"
    is_bf16_supported_fn = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported_fn) and is_bf16_supported_fn():
        return "bf16-mixed"
    return "32-true"


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


def _resolve_training_preprocessing(dataset, train_idx):
    imagery_type = dataset.imagery_type
    if imagery_type != "vhr":
        return imagery_type, None, None

    vhr_min_values, vhr_max_values = estimate_vhr_minmax_from_dataset(dataset, sample_indices=train_idx)
    print("Estimated VHR preprocessing bounds from cropped training patches.")
    return imagery_type, vhr_min_values.tolist(), vhr_max_values.tolist()


def _resolve_experiment_prefix(imagery_type):
    if imagery_type == "orthophoto":
        return "ortho"
    if imagery_type == "vhr":
        return "vhr"
    return "exp"


def _next_experiment_dir(log_root, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}_exp(\d+)$")
    max_index = 0
    for child in log_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return log_root / f"{prefix}_exp{max_index + 1}"


def _create_loggers(args):
    if args.experiment_dir:
        log_root = Path(args.experiment_dir)
    else:
        log_root = DEFAULT_LOG_ROOT

    log_root.mkdir(parents=True, exist_ok=True)
    prefix = _resolve_experiment_prefix(getattr(args, "imagery_type", "auto"))
    log_dir = _next_experiment_dir(log_root, prefix)
    log_dir.mkdir(parents=True, exist_ok=False)

    tensorboard_logger = TensorBoardLogger(save_dir=str(log_dir), name="", version="")
    csv_logger = CSVLogger(save_dir=str(log_dir), name="", version="")
    return tensorboard_logger, csv_logger, log_dir


def _load_checkpoint_state(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model


def _checkpoint_has_trainer_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return False
    return "optimizer_states" in checkpoint or "lr_schedulers" in checkpoint or "loops" in checkpoint


def _prepare_resume_checkpoint(model, checkpoint_path):
    if not checkpoint_path:
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if _checkpoint_has_trainer_state(checkpoint):
        print(f"Resuming full training state from: {checkpoint_path}")
        return checkpoint_path

    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from: {checkpoint_path}")
    print("Optimizer and scheduler state were not found, so training will warm-start from epoch 0.")
    return None


def _loader_worker_kwargs(num_workers):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 1
    return kwargs


def _export_best_validation_artifacts(
    checkpoint_path,
    dataset,
    val_indices,
    batch_size,
    num_workers,
    num_classes,
    lr,
    warmup_epochs,
    save_dir,
):
    if not checkpoint_path:
        print("Skipping best-validation export because no best checkpoint was saved.")
        return

    subset = torch.utils.data.Subset(dataset, val_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        **_loader_worker_kwargs(num_workers),
    )

    model = get_model(
        num_classes=num_classes,
        lr=lr,
        save_dir=str(save_dir),
        warmup_epochs=warmup_epochs,
    )
    model = _load_checkpoint_state(model, checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    all_targets = []
    all_preds = []
    total_loss = 0.0
    total_samples = 0
    seen_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            batch_size_actual = labels.size(0)
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

            batch_targets = labels.cpu().tolist()
            batch_preds = preds.cpu().tolist()
            all_targets.extend(batch_targets)
            all_preds.extend(batch_preds)

            probs_cpu = probs.cpu().numpy()
            for sample_offset in range(batch_size_actual):
                global_idx = val_indices[seen_samples + sample_offset]
                _image_path, image_id = dataset.image_files[global_idx]
                prob_row = probs_cpu[sample_offset]
                prediction_row = {
                    "image_id": image_id,
                    "true_label": int(batch_targets[sample_offset]),
                    "predicted_label": int(batch_preds[sample_offset]),
                    "top1_label": int(batch_preds[sample_offset]),
                    "top1_prob": float(prob_row.max()),
                }
                prediction_row.update({f"p{i}": float(prob) for i, prob in enumerate(prob_row)})
                predictions.append(prediction_row)

            seen_samples += batch_size_actual

    pred_df = pd.DataFrame(predictions)
    pred_path = Path(save_dir) / "best_val_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    oa = accuracy_score(all_targets, all_preds)
    aa = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    mean_loss = total_loss / total_samples if total_samples else 0.0
    report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=3, zero_division=0)

    report_path = Path(save_dir) / "best_val_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    cm = confusion_matrix(all_targets, all_preds)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_path = Path(save_dir) / "best_val_confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    summary_path = Path(save_dir) / "best_val_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"checkpoint_path: {checkpoint_path}\n")
        f.write(f"num_samples: {total_samples}\n")
        f.write(f"loss: {mean_loss:.6f}\n")
        f.write(f"overall_accuracy: {oa:.6f}\n")
        f.write(f"average_accuracy: {aa:.6f}\n")
        f.write(f"macro_f1: {f1:.6f}\n")

    print(f"Best-checkpoint validation predictions saved to {pred_path}")
    print(f"Best-checkpoint validation report saved to {report_path}")
    print(f"Best-checkpoint validation confusion matrix saved to {cm_path}")
    print(f"Best-checkpoint validation summary saved to {summary_path}")


def run_training(args):
    num_workers = getattr(args, "num_workers", 2)
    dataset_kwargs = _build_dataset_kwargs(args)
    image_dir_lower = str(args.image_dir).lower()
    requested_vhr = getattr(args, "imagery_type", "auto") == "vhr" or (
        getattr(args, "imagery_type", "auto") == "auto" and "vhr" in image_dir_lower
    )
    if requested_vhr and getattr(args, "vhr_min_values", None) is None and getattr(args, "vhr_max_values", None) is None:
        dataset_kwargs["resolve_vhr_minmax"] = False

    dataset = LucasDataset(args.image_dir, args.csv_path, **dict(dataset_kwargs, verbose=False))
    labels = [dataset.id_to_label[lucas_id] for (_path, lucas_id) in dataset.image_files]
    indices = list(range(len(dataset.image_files)))

    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=temp_labels)

    imagery_type, vhr_min_values, vhr_max_values = _resolve_training_preprocessing(dataset, train_idx)
    args.imagery_type = imagery_type
    args.vhr_min_values = vhr_min_values
    args.vhr_max_values = vhr_max_values
    dataset_kwargs = _build_dataset_kwargs(args)
    dataset_kwargs["imagery_type"] = imagery_type
    if vhr_min_values is not None:
        dataset_kwargs["vhr_min_values"] = vhr_min_values
        dataset_kwargs["vhr_max_values"] = vhr_max_values

    train_dataset = LucasDataset(
        args.image_dir,
        args.csv_path,
        **dict(dataset_kwargs, transform=train_transform(), verbose=False),
    )
    val_dataset = LucasDataset(
        args.image_dir,
        args.csv_path,
        **dict(dataset_kwargs, transform=val_transform(), verbose=False),
    )

    train_ds = torch.utils.data.Subset(train_dataset, train_idx)
    val_ds = torch.utils.data.Subset(val_dataset, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **_loader_worker_kwargs(num_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        **_loader_worker_kwargs(num_workers),
    )

    print(f"Train set size: {len(train_ds)} ({len(train_ds)/len(dataset)*100:.1f}%)")
    print(f"Validation set size: {len(val_ds)} ({len(val_ds)/len(dataset)*100:.1f}%)")
    print(f"Test set size: {len(test_idx)} ({len(test_idx)/len(dataset)*100:.1f}%)")

    logger, csv_logger, log_dir = _create_loggers(args)
    print(f"Logging to directory: {log_dir}")

    num_classes = dataset.num_classes if hasattr(dataset, "num_classes") else len(set(labels))
    model = get_model(
        num_classes=num_classes,
        lr=args.lr,
        save_dir=str(log_dir),
        warmup_epochs=args.warmup_epochs,
    )
    model.image_dir = args.image_dir
    model.num_samples_train = len(train_ds)
    model.num_samples_val = len(val_ds)
    model.num_samples_test = len(test_idx)
    model.num_epochs = args.max_epochs
    model.val_image_ids = [dataset.image_files[idx][1] for idx in val_idx]
    model.test_image_ids = [dataset.image_files[idx][1] for idx in test_idx]

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0005,
        patience=5,
        verbose=True,
        mode='min'
    )

    lr_monitor = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        save_weights_only=SAVE_WEIGHTS_ONLY_CHECKPOINTS,
    )
    trainer_precision = _resolve_trainer_precision()
    print(f"Lightning precision: {trainer_precision}")
    print(f"Checkpoint directory: {logger.log_dir}")
    if SAVE_WEIGHTS_ONLY_CHECKPOINTS:
        print("Checkpoint mode: weights-only Lightning .ckpt files")

    fit_ckpt_path = _prepare_resume_checkpoint(model, args.resume_from_checkpoint)

    trainer = Trainer(
        max_epochs=args.max_epochs, 
        logger=[logger, csv_logger], 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=trainer_precision,
        callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
        default_root_dir=str(log_dir),
        limit_train_batches=0.25,
        limit_test_batches=0.25,
        limit_val_batches=0.25,
        limit_predict_batches=0.25,
    )
    
    trainer.fit(model, train_loader, val_loader, ckpt_path=fit_ckpt_path)
    print(f"\nTraining completed. Metrics and reports are saved in {log_dir}")
    if checkpoint_callback.best_model_path:
        print(f"Best checkpoint saved to: {checkpoint_callback.best_model_path}")
    if checkpoint_callback.last_model_path:
        print(f"Last checkpoint saved to: {checkpoint_callback.last_model_path}")

    try:
        _save_split_ids(log_dir, dataset, train_idx, val_idx, test_idx)
        print(f"Train/val/test image_ids saved to {log_dir}")
    except Exception as e:
        print(f"Could not save train/val/test image_ids: {e}")

    config_path = os.path.join(log_dir, "config.json")
    try:
        _save_experiment_config(
            config_path=config_path,
            args=args,
            train_count=len(train_ds),
            val_count=len(val_ds),
            test_count=len(test_idx),
            log_dir=log_dir,
            num_workers=num_workers,
            trainer_precision=trainer_precision,
            num_classes=num_classes,
            preprocessing_summary=train_dataset.preprocessing_summary,
            best_checkpoint_path=checkpoint_callback.best_model_path,
            last_checkpoint_path=checkpoint_callback.last_model_path,
        )
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Could not save configuration: {e}")

    try:
        _export_best_validation_artifacts(
            checkpoint_path=checkpoint_callback.best_model_path,
            dataset=val_dataset,
            val_indices=val_idx,
            batch_size=args.batch_size,
            num_workers=0,
            num_classes=num_classes,
            lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            save_dir=log_dir,
        )
    except Exception as e:
        print(f"Could not export best-validation artifacts: {e}")
