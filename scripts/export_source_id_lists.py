import argparse
import csv
import os
import re
from pathlib import Path


CSV_ENCODINGS = ("utf-8", "iso-8859-1", "windows-1252")
ALLOWED_IMAGE_EXTENSIONS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp")


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


def normalize_id(raw_id):
    value = str(raw_id).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value


def extract_lucas_id(filename):
    match = re.search(r"ID-(\d+)", filename)
    if match:
        return match.group(1)
    match = re.search(r"(\d{5,})", filename)
    if match:
        return match.group(1)
    return None


def load_label_ids(csv_path):
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            with open(csv_path, "r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError(f"No header found in CSV: {csv_path}")
                if "IDPOINT" in reader.fieldnames:
                    id_column = "IDPOINT"
                elif "lucasId" in reader.fieldnames:
                    id_column = "lucasId"
                else:
                    raise ValueError("CSV must contain either IDPOINT or lucasId.")

                label_ids = set()
                for row in reader:
                    if row.get(id_column) is None:
                        continue
                    normalized = normalize_id(row[id_column])
                    if normalized:
                        label_ids.add(normalized)
                return label_ids
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to read CSV IDs from {csv_path}")


def discover_ids(image_dir):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    discovered = set()
    for root, dirnames, filenames in os.walk(image_dir):
        dirnames.sort()
        for filename in sorted(filenames):
            if not filename.lower().endswith(ALLOWED_IMAGE_EXTENSIONS):
                continue
            lucas_id = extract_lucas_id(filename)
            if lucas_id:
                discovered.add(lucas_id)
    return discovered


def write_ids(ids_set, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for lucas_id in sorted(ids_set):
            f.write(f"{lucas_id}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export orthophoto/VHR ID allowlists so mixed image folders can be filtered safely."
    )
    parser.add_argument("--ortho_dir", required=True, help="Directory containing orthophoto files")
    parser.add_argument("--vhr_dir", required=True, help="Directory containing VHR files")
    parser.add_argument("--csv_path", default=str(resolve_default_csv_path()), help="Path to labels CSV")
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parent / "source_id_lists"),
        help="Output directory for exported txt files",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)

    label_ids = load_label_ids(csv_path)
    ortho_ids = discover_ids(args.ortho_dir)
    vhr_ids = discover_ids(args.vhr_dir)

    ortho_ids = ortho_ids.intersection(label_ids)
    vhr_ids = vhr_ids.intersection(label_ids)

    ortho_out = output_dir / "orthophoto_ids.txt"
    vhr_out = output_dir / "vhr_ids.txt"
    write_ids(ortho_ids, ortho_out)
    write_ids(vhr_ids, vhr_out)

    print(f"CSV label IDs: {len(label_ids)}")
    print(f"Orthophoto IDs exported: {len(ortho_ids)} -> {ortho_out}")
    print(f"VHR IDs exported: {len(vhr_ids)} -> {vhr_out}")


if __name__ == "__main__":
    main()
