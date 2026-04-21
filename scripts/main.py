import argparse
from pathlib import Path

import lightning
import torch

from test import run_testing
from train import run_training


def _resolve_default_csv_path():
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "csv" / "LUCAS-Master_2025_v6.csv",
        repo_root / "LUCAS-Master_2025_v6.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_CSV_PATH = _resolve_default_csv_path()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightning Land Cover Classification")
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Training or testing mode")
    parser.add_argument("--csv_path", default=str(DEFAULT_CSV_PATH), help="Path to CSV file")
    parser.add_argument("--image_dir", required=True, help="Path to folder with images")
    parser.add_argument("--checkpoint_path", help="Path to model checkpoint (required for test mode)")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Number of warmup epochs for learning rate scheduling")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (set 0 on low-RAM systems)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patch_meters", type=int, default=384, help="Center patch size in meters")
    parser.add_argument("--imagery_type", choices=["auto", "orthophoto", "vhr"], default="auto",
                        help="Image-source preprocessing preset. Use 'orthophoto' or 'vhr' for explicit control.")
    parser.add_argument("--filter_points", action="store_true", help="Filter to use only images where interpreters STRATA1_S1 and STRATA1_S2 agree")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help=(
            "Train mode: optional custom log root (default is F:\\mixalis_projects\\lucas\\lucas_logs). "
            "Test mode: required experiment directory containing config/checkpoints."
        ),
    )
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume training from")
    parser.add_argument('--crop_mode', type=str, default='center_crop', choices=['center_crop', 'none'],
                        help="'center_crop' for patch/crop classification, 'none' for full image classification")
    parser.add_argument("--save_predictions", action="store_true", 
                        help="Save detailed predictions to CSV")

    args = parser.parse_args()

    lightning.seed_everything(42, workers=True)
    torch.backends.cudnn.allow_tf32 = True
    # Matrix Multiplications
    torch.set_float32_matmul_precision(
        # Use TF32 or 2 x BF16 if an appropriate algorithm is available and the device supports it.
        "high"
    )
    if args.mode == "train":
        run_training(args)
    elif args.mode == "test":
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required for test mode")
        if not args.experiment_dir:
            raise ValueError("--experiment_dir is required for test mode")
        run_testing(args)
