import os
import torch
from model_1 import get_model
from dataset import LucasDataset
from torch.utils.data import DataLoader
import pandas as pd
import re
from torchvision import transforms as T
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import csv


#TO BE WRITTE CORRECTLY







# Human-readable class descriptions (used for annotated images)
CLASS_DESCRIPTIONS = {
    0: "Arable land",
    1: "Permanent crops",
    2: "Grassland",
    3: "Wooded areas",
    4: "Shrubs",
    5: "Bare surface, low or rare vegetation",
    6: "Artificial constructions and sealed areas",
    7: "Inland waters / Transitional waters and Coastal waters"
}

try:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
except Exception:
    rasterio = None

try:
    import cv2
except Exception:
    cv2 = None


def run_inference(model, loader, dataset):
    model.eval()
    device = next(model.parameters()).device
    rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference"):
            images = batch["image"].to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            # For each sample in the batch, extract top-2
            for i in range(probs.shape[0]):
                prob_row = probs[i]
                idxs = prob_row.argsort()[::-1]
                top1 = int(idxs[0])
                # report confidence as percent (not raw probability)
                top1_conf = float(prob_row[top1]) * 100.0
                top2 = int(idxs[1]) if probs.shape[1] > 1 else None
                top2_conf = float(prob_row[top2]) * 100.0 if top2 is not None else None

                image_id = batch["image_id"][i]
                row = {
                    "image_id": image_id,
                    "top1_label": top1,
                    "top1_confidence": top1_conf,
                    "top2_label": top2,
                    "top2_confidence": top2_conf,
                }
                rows.append(row)
                # If a writer was provided, flush this row to the main CSV and the per-subdir CSV
                try:
                    writer = globals().get('_INFERENCE_CSV_WRITER', None)
                    if writer is not None:
                        writer.add_row(row)
                except Exception:
                    # Non-fatal: continue collecting rows even if writer fails briefly
                    pass

    df = pd.DataFrame(rows)
    return df


def annotate_and_save(image_path, top1_label, top1_conf, top2_label, top2_conf, out_path):
    # Try rasterio first for geotiffs, otherwise cv2
    img = None
    if rasterio is not None:
        try:
            with rasterio.open(image_path) as src:
                cnt = src.count
                if cnt >= 3:
                    arr = src.read([1, 2, 3]).transpose(1, 2, 0)
                elif cnt == 2:
                    a = src.read(1); b = src.read(2); arr = np.stack([a, b, a], axis=2)
                elif cnt == 1:
                    a = src.read(1); arr = np.stack([a, a, a], axis=2)
                else:
                    raise RuntimeError("Unexpected band count")
                # Normalize to 0-255
                arr = arr.astype(np.float32)
                # percentile clamp
                lo, hi = np.percentile(arr, (1, 99))
                arr = np.clip((arr - lo) / max((hi - lo), 1e-6), 0, 1)
                img = (arr * 255).astype(np.uint8)
        except Exception:
            img = None

    if img is None and cv2 is not None:
        try:
            bgr = cv2.imread(image_path)
            if bgr is not None:
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            img = None

    if img is None:
        # Create a blank placeholder
        img = np.zeros((224, 224, 3), dtype=np.uint8) + 128

    # Draw text onto image using cv2 if available — adaptively scale font to fit
    try:
        import cv2 as _cv
        out = img.copy()
        h, w = out.shape[:2]
        font = _cv.FONT_HERSHEY_SIMPLEX
        thickness = max(1, int(max(w, h) / 500))

        # Prepare compact lines: only top1 and top2 with numeric label = human-readable name
        lines = []
        name1 = CLASS_DESCRIPTIONS.get(top1_label, "")
        lines.append(f"TOP1: {top1_label} = {name1} ({top1_conf:.1f}%)")
        if top2_label is not None:
            name2 = CLASS_DESCRIPTIONS.get(top2_label, "")
            lines.append(f"TOP2: {top2_label} = {name2} ({top2_conf:.1f}%)")
        else:
            lines.append("TOP2: N/A")

        # Compute maximum allowed width for text block (90% of image width)
        pad_x = max(8, int(w * 0.02))
        pad_y = max(6, int(h * 0.01))
        max_text_width = int(w * 0.9) - 2 * pad_x

        # Find scale that makes the longest line fit horizontally
        # measure at scale=1.0 then scale down as needed
        longest_line = max(lines, key=lambda s: len(s)) if lines else ""
        try:
            (tw1, th1), _ = _cv.getTextSize(longest_line, font, 1.0, thickness)
            if tw1 <= 0:
                base_scale = 0.6
            else:
                base_scale = min(1.2, max(0.3, (max_text_width / tw1)))
        except Exception:
            base_scale = max(0.35, min(1.0, w / 800.0))

        # Also cap scale relative to image size to keep text readable
        max_scale_by_size = max(0.35, min(1.8, w / 600.0))
        font_scale = min(base_scale, max_scale_by_size)

        # Measure height for all lines at this scale
        total_h = 0
        line_sizes = []
        for line in lines:
            (tw, th), baseline = _cv.getTextSize(line, font, font_scale, thickness)
            line_sizes.append((tw, th + baseline))
            total_h += th + baseline + 6

        # determine rectangle placement bottom-right
        rect_w = min(int(max_text_width + 2 * pad_x), w - 10)
        rect_h = min(int(total_h + 2 * pad_y), h - 10)
        x_right = w - 5
        y_bottom = h - 5
        left = max(5, x_right - rect_w)
        top = max(5, y_bottom - rect_h)
        right = x_right
        bottom = y_bottom

        # Draw semi-transparent background
        try:
            overlay = out.copy()
            _cv.rectangle(overlay, (left, top), (right, bottom), (0, 0, 0), -1)
            alpha = 0.55
            _cv.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
        except Exception:
            pass

        # Render lines inside the rectangle with small padding
        x_text = left + pad_x
        y_text = top + pad_y + 2
        for i, line in enumerate(lines):
            _cv.putText(out, line, (x_text, y_text + line_sizes[i][1] - 4), font, font_scale, (255, 255, 255), thickness, _cv.LINE_AA)
            y_text += line_sizes[i][1] + 6

        # Convert back to BGR for saving via cv2
        out_bgr = _cv.cvtColor(out, _cv.COLOR_RGB2BGR)
        _cv.imwrite(out_path, out_bgr)
        return
    except Exception:
        # Fall through to PIL-based fallback
        pass

    # Fallback: save using numpy + PIL if cv2 not available
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        name1 = CLASS_DESCRIPTIONS.get(top1_label, "")
        line1 = f"TOP1: {top1_label} = {name1} ({top1_conf:.1f}%)"
        if top2_label is not None:
            name2 = CLASS_DESCRIPTIONS.get(top2_label, "")
            line2 = f"TOP2: {top2_label} = {name2} ({top2_conf:.1f}%)"
        else:
            line2 = "TOP2: N/A"
        lines = [line1, line2]

        w_img, h_img = pil.size
        pad_x = max(8, int(w_img * 0.02))
        pad_y = max(6, int(h_img * 0.01))
        max_text_width = int(w_img * 0.9) - 2 * pad_x

        # Try to use a truetype font scaled to image size; fallback to default
        truetype_font = None
        font_size = max(10, int(w_img / 40))
        try:
            # Common on Windows; if not available this will raise
            truetype_font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            try:
                truetype_font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except Exception:
                truetype_font = None

        if truetype_font is not None:
            # shrink font_size until longest line fits
            longest = max(lines, key=lambda s: len(s))
            w_long, h_long = draw.textsize(longest, font=truetype_font)
            if w_long > max_text_width and w_long > 0:
                scale = max_text_width / float(w_long)
                font_size = max(8, int(font_size * scale))
                try:
                    truetype_font = ImageFont.truetype(truetype_font.path, font_size)
                except Exception:
                    truetype_font = ImageFont.truetype("arial.ttf", font_size) if truetype_font else ImageFont.load_default()
            font = truetype_font
        else:
            font = ImageFont.load_default()

        # measure block height
        line_heights = [draw.textsize(line, font=font)[1] for line in lines]
        total_h = sum(line_heights) + (len(lines) - 1) * 6

        text_block_width = min(max_text_width, max((draw.textsize(line, font=font)[0] for line in lines)))
        text_block_height = total_h
        left = max(5, w_img - int(text_block_width) - 2 * pad_x - 10)
        top = max(5, h_img - int(text_block_height) - 2 * pad_y - 10)
        right = w_img - 10
        bottom = h_img - 10

        try:
            draw.rectangle([left, top, right, bottom], fill=(0, 0, 0))
        except Exception:
            pass

        tx = left + pad_x
        ty = top + pad_y
        for i, line in enumerate(lines):
            draw.text((tx, ty), line, fill=(255, 255, 255), font=font)
            ty += line_heights[i] + 6

        pil.save(out_path)
        return
    except Exception:
        # give up silently
        return


class SimpleImageDataset(torch.utils.data.Dataset):
    """Simple dataset for unlabeled image folders. Extracts an ID from filename and returns a transformed tensor and image_id."""
    def __init__(self, image_dir, output_size=(224, 224)):
        self.image_dir = image_dir
        allowed_exts = ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp')
        # Walk the directory tree to find images in nested subfolders (preserve subfolder structure)
        self.paths = []
        self.files = []  # store relative path from image_dir
        for root, _, filenames in os.walk(image_dir):
            for fname in sorted(filenames):
                if fname.lower().endswith(allowed_exts):
                    full = os.path.join(root, fname)
                    rel = os.path.relpath(full, image_dir)
                    self.paths.append(full)
                    self.files.append(rel)
        # IDs extracted from basename of files
        self.ids = [self._extract_id(os.path.basename(f)) for f in self.files]
        # transforms: resize -> to tensor -> normalize (ImageNet stats)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(output_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _extract_id(self, fname):
        # mirror LucasDataset id extraction rules
        m = re.search(r'ID-(\d+)', fname)
        if m:
            return m.group(1)
        m2 = re.search(r'(\d{5,})', fname)
        if m2:
            return m2.group(1)
        # fallback to filename without ext
        return os.path.splitext(fname)[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = None
        # Try rasterio then cv2
        if rasterio is not None:
            try:
                with rasterio.open(path) as src:
                    cnt = src.count
                    if cnt >= 3:
                        arr = src.read([1, 2, 3]).transpose(1, 2, 0)
                    elif cnt == 2:
                        a = src.read(1); b = src.read(2); arr = np.stack([a, b, a], axis=2)
                    elif cnt == 1:
                        a = src.read(1); arr = np.stack([a, a, a], axis=2)
                    else:
                        raise RuntimeError("Unexpected band count")
                    arr = arr.astype(np.float32)
                    lo, hi = np.percentile(arr, (1, 99))
                    arr = np.clip((arr - lo) / max((hi - lo), 1e-6), 0, 1)
                    img = (arr * 255).astype(np.uint8)
            except Exception:
                img = None
        if img is None and cv2 is not None:
            try:
                bgr = cv2.imread(path)
                if bgr is not None:
                    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                img = None
        if img is None:
            # fallback blank
            img = np.zeros((224, 224, 3), dtype=np.uint8) + 128
        tensor = self.transform(img)
        return {"image": tensor, "image_id": self.ids[idx]}


class CsvProgressWriter:
    """Incrementally writes a main predictions CSV and per-subfolder CSVs as rows arrive.

    Usage: create before inference with the final output paths and pass rows via add_row(dict).
    """
    def __init__(self, out_csv, image_dir, out_root, id_to_path):
        self.out_csv = out_csv
        self.image_dir = image_dir
        self.out_root = out_root
        # normalize id keys to strings for robust matching
        try:
            self.id_to_path = {str(k): v for k, v in id_to_path.items()}
        except Exception:
            self.id_to_path = {}

        # ensure parent exists
        parent = os.path.dirname(out_csv) or '.'
        os.makedirs(parent, exist_ok=True)

        # Header columns expected by callers
        self.columns = ['IDPOINT', 'top1_label', 'top1_confidence', 'top2_label', 'top2_confidence']
        # Create / overwrite main CSV with header
        try:
            pd.DataFrame(columns=self.columns).to_csv(self.out_csv, index=False)
        except Exception:
            # fallback: try simple open
            try:
                with open(self.out_csv, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.columns)
            except Exception:
                pass

        # track which sub-csvs have been created to avoid rewriting header
        self._created_subcsvs = set()

    def add_row(self, row):
        # map row format to expected columns
        out_row = {
            'IDPOINT': str(row.get('image_id')),
            'top1_label': row.get('top1_label'),
            'top1_confidence': row.get('top1_confidence'),
            'top2_label': row.get('top2_label'),
            'top2_confidence': row.get('top2_confidence')
        }

        # append to main CSV
        try:
            pd.DataFrame([out_row]).to_csv(self.out_csv, mode='a', header=False, index=False)
        except Exception:
            # best-effort fallback
            try:
                with open(self.out_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([out_row[c] for c in self.columns])
            except Exception:
                pass

        # write to per-subfolder CSV if we can determine the source path
        img_id = out_row['IDPOINT']
        if img_id in self.id_to_path:
            src = self.id_to_path[img_id]
            try:
                src_norm = os.path.normpath(src)
                image_dir_norm = os.path.normpath(self.image_dir)
                rel_dir = os.path.relpath(os.path.dirname(src_norm), image_dir_norm)
            except Exception:
                rel_dir = os.path.basename(os.path.dirname(src))

            if rel_dir in ('.', ''):
                subdir = self.out_root
                # use the image_dir basename as folder name for root-level files
                folder_name = os.path.basename(os.path.normpath(self.image_dir)) or 'root'
            else:
                subdir = os.path.join(self.out_root, rel_dir)
                # use only the last path component as the folder name
                folder_name = os.path.basename(rel_dir) or rel_dir.replace(os.sep, '_')
            # sanitize folder_name to avoid problematic characters
            folder_name = folder_name.replace(' ', '_')
            os.makedirs(subdir, exist_ok=True)
            sub_csv = os.path.join(subdir, f'predictions_{folder_name}.csv')

            # create header if needed
            if sub_csv not in self._created_subcsvs:
                try:
                    pd.DataFrame(columns=self.columns).to_csv(sub_csv, index=False)
                    self._created_subcsvs.add(sub_csv)
                except Exception:
                    try:
                        with open(sub_csv, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(self.columns)
                        self._created_subcsvs.add(sub_csv)
                    except Exception:
                        pass

            # append row to sub_csv
            try:
                pd.DataFrame([out_row]).to_csv(sub_csv, mode='a', header=False, index=False)
            except Exception:
                try:
                    with open(sub_csv, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([out_row[c] for c in self.columns])
                except Exception:
                    pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a folder of images and output top1/top2 predictions CSV")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint (.pt or state_dict)")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to directory containing images")
    parser.add_argument("--csv_path", type=str, required=False, default=None, help="(optional) Path to master CSV used by LucasDataset to match IDs. If omitted, images are treated as unlabeled and IDs are extracted from filenames.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--backbone", type=str, default="lulc_custom")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_csv", type=str, default=None, help="Path to write predictions CSV. Defaults to image_dir/predictions.csv")
    parser.add_argument("--save", action="store_true", help="Also save annotated images with top1/top2 text into output_dir")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save annotated images when --save is used")
    args = parser.parse_args()

    # Load the checkpoint early so we can infer num_classes if needed
    print(f"Loading checkpoint metadata from: {args.checkpoint_path}")
    raw_checkpoint = torch.load(args.checkpoint_path, map_location='cpu')

    def _extract_state_dict(ckpt):
        if isinstance(ckpt, dict):
            # common Lightning style
            if 'state_dict' in ckpt:
                return ckpt['state_dict']
            # sometimes saved with 'model' prefix
            for k in ('model', 'state', 'weights'):
                if k in ckpt and isinstance(ckpt[k], dict):
                    return ckpt[k]
            # otherwise assume the dict itself is a state_dict-like mapping
            return {k: v for k, v in ckpt.items() if hasattr(v, 'ndim')}
        return None

    state_dict = _extract_state_dict(raw_checkpoint)

    def infer_num_classes_from_state_dict(sd):
        if sd is None:
            return None
        candidates = []
        for k, v in sd.items():
            try:
                shape = tuple(v.shape)
            except Exception:
                continue
            # look for 2D weight tensors likely to be classifier weights
            if len(shape) == 2:
                # prefer keys that mention 'fc' (classifier) or 'classifier'
                key_lower = k.lower()
                if '.fc.' in key_lower or 'fc.' in key_lower or 'classifier' in key_lower:
                    candidates.append((k, shape))
        # prefer candidate where hidden dim == 256 (the LULC_Model uses 256 in penultimate layer)
        for k, shape in candidates:
            if shape[1] == 256:
                return shape[0]
        # fallback: choose the candidate with smallest output dim (>1)
        small_candidates = [shape[0] for _, shape in candidates if shape[0] > 1]
        if small_candidates:
            return min(small_candidates)
        # last resort: try any 2D weight with small first dim
        other = []
        for k, v in sd.items():
            try:
                shape = tuple(v.shape)
            except Exception:
                continue
            if len(shape) == 2 and shape[0] > 1 and shape[0] < 1024:
                other.append(shape[0])
        if other:
            return min(other)
        return None

    inferred_num_classes = infer_num_classes_from_state_dict(state_dict)
    if inferred_num_classes is not None:
        print(f"Inferred num_classes from checkpoint: {inferred_num_classes}")
    else:
        print("Could not infer num_classes from checkpoint; will try to get it from CSV if provided.")

    # Load dataset: either labeled via CSV (LucasDataset) or unlabeled image folder
    if args.csv_path:
        dataset = LucasDataset(args.image_dir, args.csv_path, excluded_classes=[10], crop_mode="none") #crop_mode maybe change
        print(f"Loaded dataset with {len(dataset)} candidates (images matching CSV IDs)")
        if len(dataset) == 0:
            print("No images found that match IDs in CSV. Exiting.")
            raise SystemExit(1)
        num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else None
        if num_classes is None and inferred_num_classes is not None:
            num_classes = inferred_num_classes
    else:
        # Unlabeled images: use the top-level SimpleImageDataset which is picklable by multiprocessing
        dataset = SimpleImageDataset(args.image_dir)
        print(f"Loaded {len(dataset)} unlabeled images from {args.image_dir}")
        if len(dataset) == 0:
            print("No images found in image_dir. Exiting.")
            raise SystemExit(1)
        # If checkpoint provided num_classes inference, use it
        if inferred_num_classes is not None:
            num_classes = inferred_num_classes
        else:
            print("Could not infer num_classes from checkpoint; please provide a labeled CSV so num_classes can be determined. Exiting.")
            raise SystemExit(1)

    # Prepare output CSV paths and incremental writer BEFORE running inference so progress is visible
    out_csv = args.output_csv or os.path.join(args.image_dir, 'predictions_inference.csv')
    # If the user passed a directory by mistake, write the default filename inside it.
    if os.path.isdir(out_csv) or str(out_csv).endswith(os.path.sep):
        out_csv = os.path.join(out_csv, 'predictions_inference.csv')
    try:
        parent = os.path.dirname(out_csv) or '.'
        os.makedirs(parent, exist_ok=True)
    except Exception:
        pass

    # Determine root folder where per-subfolder CSVs / annotated images will be written
    out_root = args.output_dir if getattr(args, 'output_dir', None) else (os.path.dirname(out_csv) or args.image_dir)

    # Build id -> src path map (works for both labeled and unlabeled dataset types)
    id_to_path = {}
    if args.csv_path and hasattr(dataset, 'image_files'):
        try:
            id_to_path = {str(img_id): path for (path, img_id) in dataset.image_files}
        except Exception:
            id_to_path = {}
    else:
        try:
            id_to_path = {str(dataset.ids[i]): dataset.paths[i] for i in range(len(dataset))}
        except Exception:
            id_to_path = {}

    # Instantiate incremental CSV writer and register globally for run_inference to use
    try:
        _INFERENCE_CSV_WRITER = CsvProgressWriter(out_csv, args.image_dir, out_root, id_to_path)
        globals()['_INFERENCE_CSV_WRITER'] = _INFERENCE_CSV_WRITER
    except Exception:
        _INFERENCE_CSV_WRITER = None

    # Create loader if not created in unlabeled branch
    if 'loader' not in locals():
        # choose a safe default for num_workers on Windows
        try:
            cpu_count = os.cpu_count() or 1
            num_workers = min(4, max(0, cpu_count - 1))
        except Exception:
            num_workers = 16
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=4)

    # Model
    model = get_model(num_classes, backbone=args.backbone, lr=args.lr, save_dir=None)

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    pred_df = run_inference(model, loader, dataset)

    # Finalize / report
    try:
        print(f"Predictions saved (incrementally) to: {out_csv}")
    except Exception:
        pass

    # Optionally save annotated images
    if args.save:
        # Use same out_root as above so annotated images and per-subfolder CSVs live together
        out_dir = out_root if 'out_root' in locals() else (args.output_dir or os.path.join(args.image_dir, 'annotated'))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # map image_id -> image path for both dataset types
        # reuse the normalized id_to_path we built earlier if present, else build a string-keyed map
        try:
            id_to_path = globals().get('_INFERENCE_CSV_WRITER').id_to_path if globals().get('_INFERENCE_CSV_WRITER') is not None else {}
        except Exception:
            id_to_path = {}
        if not id_to_path:
            if args.csv_path and hasattr(dataset, 'image_files'):
                try:
                    id_to_path = {str(img_id): path for (path, img_id) in dataset.image_files}
                except Exception:
                    id_to_path = {}
            else:
                try:
                    id_to_path = {str(dataset.ids[i]): dataset.paths[i] for i in range(len(dataset))}
                except Exception:
                    id_to_path = {}

        for _, row in pred_df.iterrows():
            img_id = str(row['image_id'])
            if img_id in id_to_path:
                src = id_to_path[img_id]
                # preserve subfolder structure under image_dir when saving to out_dir
                src_norm = os.path.normpath(src)
                image_dir_norm = os.path.normpath(args.image_dir)
                try:
                    rel_dir = os.path.relpath(os.path.dirname(src_norm), image_dir_norm)
                except Exception:
                    # cross-drive relpath may fail on Windows; fallback to immediate parent folder name
                    rel_dir = os.path.basename(os.path.dirname(src_norm))

                if rel_dir in (".", ""):
                    out_subdir = out_dir
                else:
                    out_subdir = os.path.join(out_dir, rel_dir)
                os.makedirs(out_subdir, exist_ok=True)

                base = os.path.basename(src_norm)
                name, ext = os.path.splitext(base)
                if not ext:
                    ext = ".jpg"
                out_path = os.path.join(out_subdir, f"{name}_annotated{ext}")
                annotate_and_save(src, row['top1_label'], row['top1_confidence'], row['top2_label'], row['top2_confidence'], out_path)
        print(f"Annotated images saved to: {out_dir}")
