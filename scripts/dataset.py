import os
import re
import warnings
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import cv2
import h5py
import numpy as np
import pandas as pd
import rasterio
import torchvision.transforms as T
from rasterio.errors import NotGeoreferencedWarning
from rasterio.enums import Resampling
from rasterio.transform import Affine
from torch.utils.data import Dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CSV_ENCODINGS = ("utf-8", "iso-8859-1", "windows-1252")
ALLOWED_IMAGE_EXTENSIONS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp")
VHR_MINMAX_LOWER_PERCENTILE = 2.0
VHR_MINMAX_UPPER_PERCENTILE = 98.0
VHR_PER_IMAGE_LOWER_PERCENTILE = 0.5
VHR_PER_IMAGE_UPPER_PERCENTILE = 99.5
VHR_MINMAX_SAMPLE_SIZE = 512
VHR_MINMAX_PIXEL_SAMPLE_SIZE = 2048
READ_ERROR_LOG_LIMIT = 5
DEFAULT_CSV_DTYPES = {
    "IDPOINT": int,
    "X_LAEA": int,
    "Y_LAEA": int,
    "LON": float,
    "LAT": float,
    "COUNTRY": str,
    "STRATA1_S1": int,
    "STRATA2_S1": "Int64",
    "WET_S1": str,
    "ASSOC_S1": str,
    "EWO_S1": str,
    "LTNE_S1": str,
    "INTRPRT_S1": str,
    "DATE_PI_S1": str,
    "CMNT_S1": str,
    "STRATA1_S2": int,
    "STRATA2_S2": "Int64",
    "WET_S2": str,
    "ASSOC_S2": str,
    "EWO_S2": str,
    "LTNE_S2": str,
    "INTRPRT_S2": str,
    "DATE_PI_S2": str,
    "CMNT_S2": str,
    "QC": str,
    "STRATA1_S3": "Int64",
    "STRATA2_S3": "Int64",
    "WET_S3": str,
    "ASSOC_S3": str,
    "EWO_S3": str,
    "LTNE_S3": str,
    "INTRPRT_S3": str,
    "DATE_PI_S3": str,
    "CMNT_S3": str,
    "spatial_resolution": float,
    "STR25": int,
}


def train_transform(output_size=(224, 224)):
    return T.Compose([
        T.ToTensor(),
        T.Resize(output_size),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def val_transform(output_size=(224, 224)):
    return T.Compose([
        T.ToTensor(),
        T.Resize(output_size),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def test_transform(output_size=(224, 224)):
    return val_transform(output_size)


def estimate_vhr_minmax_from_dataset(
    dataset,
    image_files=None,
    sample_indices=None,
    lower_percentile=VHR_MINMAX_LOWER_PERCENTILE,
    upper_percentile=VHR_MINMAX_UPPER_PERCENTILE,
    sample_size=VHR_MINMAX_SAMPLE_SIZE,
    pixel_sample_size=VHR_MINMAX_PIXEL_SAMPLE_SIZE,
    random_seed=42,
):
    if image_files is not None and sample_indices is not None:
        raise ValueError("Provide either image_files or sample_indices, not both.")

    if image_files is None:
        if sample_indices is None:
            image_files = dataset.image_files
        else:
            image_files = [dataset.image_files[idx] for idx in sample_indices]

    if not image_files:
        raise ValueError("Cannot estimate VHR min-max bounds from an empty image list.")

    rng = np.random.default_rng(random_seed)
    sample_count = min(sample_size, len(image_files))
    sample_indices = rng.choice(len(image_files), size=sample_count, replace=False)
    sampled_pixels = [[] for _ in range(3)]

    for file_index in sample_indices:
        image_path, lucas_id = image_files[int(file_index)]
        try:
            arr = dataset._read_raw_image_with_rasterio(image_path, lucas_id).astype(np.float32)
        except Exception:
            arr = dataset._read_raw_image_with_opencv(image_path, lucas_id).astype(np.float32)

        flat = arr.reshape(3, -1).T
        num_pixels = min(pixel_sample_size, flat.shape[0])
        pixel_indices = rng.choice(flat.shape[0], size=num_pixels, replace=False)
        pixel_sample = flat[pixel_indices]
        for band_idx in range(3):
            sampled_pixels[band_idx].append(pixel_sample[:, band_idx])

    mins = []
    maxs = []
    for band_samples in sampled_pixels:
        band_values = np.concatenate(band_samples)
        mins.append(float(np.percentile(band_values, lower_percentile)))
        maxs.append(float(np.percentile(band_values, upper_percentile)))

    return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


class LucasDataset(Dataset):
    def __init__(
        self,
        image_dir,
        label_csv,
        embeddings_path=None,
        patch_meters=384,
        output_size=(224, 224),
        transform=None,
        crop_mode="center_crop",
        filter_points=False,
        csv_dtypes=None,
        imagery_type="auto",
        vhr_min_values=None,
        vhr_max_values=None,
        resolve_vhr_minmax=True,
        verbose=True,
    ):
        """
        Args:
            patch_meters: Size of center patch in meters around the observation point
            crop_mode: 'center_crop' (default, use patch/crop logic), or 'none' (use full image)
        """
        self.image_dir = image_dir
        self.patch_meters = patch_meters
        self.output_size = output_size
        self.crop_mode = crop_mode
        self.filter_points = filter_points
        self.transform = transform or val_transform(output_size)
        self.requested_imagery_type = imagery_type
        self.verbose = verbose
        self.embeddings_path = str(embeddings_path) if embeddings_path is not None else None
        self._embeddings_file = None
        self._embeddings_dataset = None
        self._embedding_dataset_key = None
        self.point_id_to_index = {}

        if self.embeddings_path is not None:
            with h5py.File(self.embeddings_path, "r") as hdf:
                self.point_ids = hdf["ids"][:]
                if "values" in hdf:
                    self._embedding_dataset_key = "values"
                elif "valus" in hdf:
                    self._embedding_dataset_key = "valus"
                else:
                    raise KeyError(f"Could not find embedding dataset in {self.embeddings_path}. Expected 'values'.")

            self.point_id_to_index = {
                self._normalize_id(point_id): index for index, point_id in enumerate(self.point_ids)
            }

        df = self._load_csv(label_csv, csv_dtypes or DEFAULT_CSV_DTYPES)
        id_column, class_column = self._resolve_csv_columns(df)
        df = self._prepare_label_dataframe(df, id_column, class_column)
        df = self._filter_interpreter_agreement(df, class_column)
        df = self._exclude_classes(df, class_column)

        unique_labels = sorted(df[class_column].unique())
        self.label_mapping = {old: new for new, old in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)

        df["mapped_label"] = df[class_column].map(self.label_mapping)
        self.id_to_label = dict(zip(df[id_column], df["mapped_label"]))

        actual_classes = sorted(df[class_column].unique())
        self._log(f"Actual classes in CSV: {actual_classes}")
        self._log(f"Number of actual classes: {len(actual_classes)}")

        self.id_to_res = self._build_resolution_mapping(df, id_column)
        self.id_to_xy = self._build_coordinate_mapping(df, id_column)
        self.image_files = self._discover_image_files(image_dir)
        self.imagery_type = self._resolve_imagery_type(imagery_type)
        self.vhr_min_values = None
        self.vhr_max_values = None
        self.preprocessing_summary = {}
        self._read_failures = 0
        self._read_failures_suppressed = False
        self._configure_preprocessing(vhr_min_values, vhr_max_values, resolve_vhr_minmax)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path, lucas_id = self.image_files[idx]
        label = self.id_to_label[lucas_id]

        try:
            img = self._read_image_with_rasterio(path, lucas_id)
        except Exception as rasterio_exc:
            try:
                img = self._read_image_with_opencv(path, lucas_id)
            except Exception as opencv_exc:
                self._log_read_failure(path, rasterio_exc, opencv_exc)
                img = np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.float32)

        img = self.transform(img)
        sample = {
            "image": img,
            "label": label,
            "image_id": lucas_id,
        }
        if self.embeddings_path is not None:
            sample["embedding"] = self._load_embedding(lucas_id)
        return sample

    def __del__(self):
        try:
            if self._embeddings_file is not None:
                self._embeddings_file.close()
        except Exception:
            pass

    def _log_read_failure(self, path, rasterio_exc, opencv_exc):
        read_failures = int(getattr(self, "_read_failures", 0)) + 1
        self._read_failures = read_failures
        if read_failures <= READ_ERROR_LOG_LIMIT:
            self._log(
                f"Image read failure ({read_failures}/{READ_ERROR_LOG_LIMIT}) for {path}\n"
                f"  rasterio: {rasterio_exc}\n"
                f"  opencv: {opencv_exc}"
            )
        elif not bool(getattr(self, "_read_failures_suppressed", False)):
            self._read_failures_suppressed = True
            self._log(
                f"Further image read failures are suppressed after {READ_ERROR_LOG_LIMIT} messages. "
                "Training will continue with zero-image fallback for unreadable samples."
            )

    def _load_csv(self, label_csv, csv_dtypes):
        try:
            return pd.read_csv(label_csv, encoding=CSV_ENCODINGS[0], dtype=csv_dtypes, low_memory=False)
        except Exception:
            for encoding in CSV_ENCODINGS[1:]:
                try:
                    return pd.read_csv(label_csv, encoding=encoding, low_memory=False)
                except UnicodeDecodeError:
                    continue
        return pd.read_csv(label_csv, low_memory=False)

    def _log(self, message):
        if bool(getattr(self, "verbose", True)):
            print(message)

    def _get_embeddings_dataset(self):
        if self.embeddings_path is None:
            raise RuntimeError("Embeddings were not configured for this dataset.")
        if self._embeddings_dataset is None:
            self._embeddings_file = h5py.File(self.embeddings_path, "r")
            self._embeddings_dataset = self._embeddings_file[self._embedding_dataset_key]
        return self._embeddings_dataset

    def _load_embedding(self, lucas_id):
        normalized_id = self._normalize_id(lucas_id)
        if normalized_id not in self.point_id_to_index:
            raise KeyError(f"Missing embedding for lucas_id={normalized_id}")

        embedding = self._get_embeddings_dataset()[self.point_id_to_index[normalized_id]]
        return np.asarray(embedding, dtype=np.float32).reshape(-1)

    def _resolve_csv_columns(self, df):
        if "IDPOINT" in df.columns and "STR25" in df.columns:
            return "IDPOINT", "STR25"
        if "lucasId" in df.columns and "STR25" in df.columns:
            return "lucasId", "STR25"
        raise ValueError("Could not resolve required label columns. Expected IDPOINT/STR25 or lucasId/STR25.")

    def _prepare_label_dataframe(self, df, id_column, class_column):
        df = df.copy()
        df[id_column] = pd.to_numeric(df[id_column], errors="coerce")
        df = df.dropna(subset=[id_column, class_column])
        df[id_column] = df[id_column].astype(int).astype(str)
        return df

    def _filter_interpreter_agreement(self, df, class_column):
        if not self.filter_points:
            return df

        original_count = len(df)
        self._log(f"Original dataset size: {original_count} images")

        if "STRATA1_S1" not in df.columns or "STRATA1_S2" not in df.columns:
            self._log("Warning: STRATA1_S1 and/or STRATA1_S2 columns not found in CSV. Ignoring --filter_points")
            return df

        class_counts_before = df[class_column].value_counts().sort_index()
        df_agreed = df[df["STRATA1_S1"] == df["STRATA1_S2"]].copy()
        filtered_count = len(df_agreed)
        removed_count = original_count - filtered_count

        self._log(f"Images where interpreters agree: {filtered_count}")
        self._log(f"Images removed due to interpreter disagreement: {removed_count}")

        self._log("\n=== Images per class BEFORE interpreter filtering ===")
        for class_val, count in class_counts_before.items():
            self._log(f"Class {class_val}: {count} images")

        self._log("\n=== Images per class AFTER interpreter filtering ===")
        class_counts_after = df_agreed[class_column].value_counts().sort_index()
        for class_val, count in class_counts_after.items():
            removed_for_class = class_counts_before.get(class_val, 0) - count
            self._log(f"Class {class_val}: {count} images (removed {removed_for_class})")

        self._log(f"\nProceeding with {len(df_agreed)} images where interpreters agree")
        return df_agreed

    def _exclude_classes(self, df, class_column):
        return df[df[class_column] != 10].copy()

    def _normalize_id(self, value):
        id_str = str(value).strip()
        if id_str.endswith(".0"):
            id_str = id_str[:-2]
        return id_str

    def _build_resolution_mapping(self, df, id_column):
        id_to_res = {}
        if "spatial_resolution" not in df.columns:
            return id_to_res

        for row_id, res_value in zip(df[id_column], df["spatial_resolution"]):
            try:
                if pd.isna(res_value):
                    continue
                id_to_res[self._normalize_id(row_id)] = float(res_value)
            except Exception:
                continue
        return id_to_res

    def _build_coordinate_mapping(self, df, id_column):
        id_to_xy = {}
        if "X_LAEA" not in df.columns or "Y_LAEA" not in df.columns:
            return id_to_xy

        for row_id, x_value, y_value in zip(df[id_column], df["X_LAEA"], df["Y_LAEA"]):
            try:
                if pd.isna(x_value) or pd.isna(y_value):
                    continue
                id_to_xy[self._normalize_id(row_id)] = (float(x_value), float(y_value))
            except Exception:
                continue
        return id_to_xy

    def _extract_lucas_id(self, filename):
        match = re.search(r"ID-(\d+)", filename)
        if match:
            return match.group(1)

        match = re.search(r"(\d{5,})", filename)
        if match:
            return match.group(1)

        return None

    def _discover_image_files(self, image_dir):
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        image_files = []
        for root, dirnames, filenames in os.walk(image_dir):
            dirnames.sort()
            for filename in sorted(filenames):
                if not filename.lower().endswith(ALLOWED_IMAGE_EXTENSIONS):
                    continue

                lucas_id = self._extract_lucas_id(filename)
                if lucas_id and lucas_id in self.id_to_label:
                    if self.embeddings_path is not None and lucas_id not in self.point_id_to_index:
                        continue
                    full_path = os.path.join(root, filename)
                    image_files.append((full_path, lucas_id))
        return image_files

    def _resolve_imagery_type(self, imagery_type):
        if imagery_type in {"orthophoto", "vhr"}:
            return imagery_type

        image_dir_lower = str(self.image_dir).lower()
        if "vhr" in image_dir_lower:
            return "vhr"
        if "ortho" in image_dir_lower:
            return "orthophoto"

        sample_resolutions = [
            self.id_to_res[lucas_id]
            for _path, lucas_id in self.image_files[:256]
            if lucas_id in self.id_to_res
        ]
        if sample_resolutions:
            median_resolution = float(np.median(sample_resolutions))
            return "vhr" if median_resolution >= 1.5 else "orthophoto"

        tif_count = sum(Path(path).suffix.lower() in {".tif", ".tiff"} for path, _lucas_id in self.image_files[:256])
        jpg_count = sum(Path(path).suffix.lower() in {".jpg", ".jpeg"} for path, _lucas_id in self.image_files[:256])
        if tif_count > jpg_count:
            return "vhr"
        return "orthophoto"

    def _configure_preprocessing(self, vhr_min_values, vhr_max_values, resolve_vhr_minmax):
        if self.imagery_type != "vhr":
            self.preprocessing_summary = {
                "imagery_type": self.imagery_type,
                "scaling": "uint8_to_unit_interval",
                "normalize_mean": IMAGENET_MEAN,
                "normalize_std": IMAGENET_STD,
            }
            self._log(f"Resolved imagery type: {self.imagery_type}")
            self._log("Using orthophoto preprocessing: uint8 scaling to [0, 1] followed by ImageNet normalization.")
            return

        if (vhr_min_values is None) != (vhr_max_values is None):
            raise ValueError("Provide both vhr_min_values and vhr_max_values together.")

        if vhr_min_values is None and resolve_vhr_minmax:
            vhr_min_values, vhr_max_values = estimate_vhr_minmax_from_dataset(self)
            source = "estimated_from_dataset_crops"
        elif vhr_min_values is None:
            source = "deferred"
        else:
            source = "explicit"

        if vhr_min_values is not None:
            self.vhr_min_values = self._coerce_band_values(vhr_min_values, "vhr_min_values")
            self.vhr_max_values = self._coerce_band_values(vhr_max_values, "vhr_max_values")
            if np.any(self.vhr_max_values <= self.vhr_min_values):
                raise ValueError("Each VHR max value must be greater than the corresponding min value.")

        self.preprocessing_summary = {
            "imagery_type": self.imagery_type,
            "scaling": "vhr_dtype_aware_scaling",
            "normalize_mean": IMAGENET_MEAN,
            "normalize_std": IMAGENET_STD,
            "vhr_uint8_scaling": "uint8_to_unit_interval",
            "vhr_high_bit_depth_scaling": "vhr_per_image_percentile_to_unit_interval",
            "vhr_min_values": self.vhr_min_values.tolist() if self.vhr_min_values is not None else None,
            "vhr_max_values": self.vhr_max_values.tolist() if self.vhr_max_values is not None else None,
            "vhr_bounds_source": source,
            "vhr_lower_percentile": VHR_MINMAX_LOWER_PERCENTILE,
            "vhr_upper_percentile": VHR_MINMAX_UPPER_PERCENTILE,
            "vhr_per_image_lower_percentile": VHR_PER_IMAGE_LOWER_PERCENTILE,
            "vhr_per_image_upper_percentile": VHR_PER_IMAGE_UPPER_PERCENTILE,
        }

        self._log(f"Resolved imagery type: {self.imagery_type}")
        if self.vhr_min_values is None:
            self._log("VHR preprocessing is configured for min-max normalization, but bounds are deferred.")
        else:
            self._log(
                "Using VHR preprocessing: dtype-aware scaling to [0, 1] followed by ImageNet normalization.\n"
                "uint8 VHR uses /255 scaling; higher-bit-depth VHR uses per-image percentile scaling.\n"
                f"VHR mins: {self.vhr_min_values.tolist()} | VHR maxs: {self.vhr_max_values.tolist()} "
                f"(source: {source})"
            )

    def _coerce_band_values(self, values, field_name):
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.shape[0] != 3:
            raise ValueError(f"{field_name} must contain exactly 3 values, got {array.shape[0]}.")
        return array

    def _resolve_crop_center(self, src, lucas_id):
        width, height = src.width, src.height
        center_x, center_y = width // 2, height // 2

        try:
            has_geo_ref = getattr(src, "crs", None) is not None and src.transform != Affine.identity()
            if lucas_id in self.id_to_xy and has_geo_ref:
                x_world, y_world = self.id_to_xy[lucas_id]
                row, col = src.index(x_world, y_world)
                return int(col), int(row)
        except Exception:
            pass

        return center_x, center_y

    def _read_rgb_bands(self, src, path, window=None, out_shape_hw=None):
        read_kwargs = {"window": window}
        if out_shape_hw is not None:
            out_h, out_w = out_shape_hw
            read_kwargs["resampling"] = Resampling.bilinear

        band_count = src.count
        if band_count >= 3:
            if out_shape_hw is not None:
                read_kwargs["out_shape"] = (3, out_h, out_w)
            return src.read([1, 2, 3], **read_kwargs).transpose(1, 2, 0)
        if band_count == 2:
            if out_shape_hw is not None:
                read_kwargs["out_shape"] = (out_h, out_w)
            band_a = src.read(1, **read_kwargs)
            band_b = src.read(2, **read_kwargs)
            return np.stack([band_a, band_b, band_a], axis=2)
        if band_count == 1:
            if out_shape_hw is not None:
                read_kwargs["out_shape"] = (out_h, out_w)
            band = src.read(1, **read_kwargs)
            return np.stack([band, band, band], axis=2)
        raise RuntimeError(f"Unexpected band count ({band_count}) in {path}")

    def _read_raw_image_with_rasterio(self, path, lucas_id):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
            with rasterio.open(path) as src:
                width, height = src.width, src.height
                center_x, center_y = self._resolve_crop_center(src, lucas_id)
                out_shape_hw = None
                if self.imagery_type == "orthophoto":
                    out_shape_hw = (self.output_size[1], self.output_size[0])

                if self.crop_mode == "none":
                    return self._read_rgb_bands(src, path, out_shape_hw=out_shape_hw)

                x0, x1, y0, y1 = self._get_crop_bounds(
                    width,
                    height,
                    center_x,
                    center_y,
                    lucas_id,
                    src.transform,
                )
                return self._read_rgb_bands(
                    src,
                    path,
                    window=((y0, y1), (x0, x1)),
                    out_shape_hw=out_shape_hw,
                )

    def _read_raw_image_with_opencv(self, path, lucas_id):
        img_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise RuntimeError("OpenCV failed to read image")

        if img_bgr.ndim == 2:
            img_rgb = np.stack([img_bgr, img_bgr, img_bgr], axis=2)
        elif img_bgr.shape[2] >= 3:
            img_rgb = cv2.cvtColor(img_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            img_rgb = np.repeat(img_bgr[:, :, :1], 3, axis=2)

        if self.crop_mode == "none":
            return img_rgb

        height, width = img_rgb.shape[:2]
        center_x, center_y = width // 2, height // 2
        x0, x1, y0, y1 = self._get_crop_bounds(
            width,
            height,
            center_x,
            center_y,
            lucas_id,
            Affine.identity(),
        )
        return img_rgb[y0:y1, x0:x1, :]

    def _scale_to_unit_interval(self, arr):
        original_dtype = arr.dtype
        arr = arr.astype(np.float32)

        if self.imagery_type == "vhr":
            if np.issubdtype(original_dtype, np.integer) and np.iinfo(original_dtype).max <= 255:
                return np.clip(arr / 255.0, 0.0, 1.0).astype(np.float32)
            mins = np.percentile(
                arr,
                VHR_PER_IMAGE_LOWER_PERCENTILE,
                axis=(0, 1),
            ).astype(np.float32).reshape(1, 1, 3)
            maxs = np.percentile(
                arr,
                VHR_PER_IMAGE_UPPER_PERCENTILE,
                axis=(0, 1),
            ).astype(np.float32).reshape(1, 1, 3)
            scale = np.maximum(maxs - mins, 1.0)
            return np.clip((arr - mins) / scale, 0.0, 1.0).astype(np.float32)

        if np.issubdtype(original_dtype, np.integer):
            scale = float(np.iinfo(original_dtype).max)
            return np.clip(arr / scale, 0.0, 1.0).astype(np.float32)

        if arr.max() > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def describe_effective_scaling(self, dtype_name):
        dtype_name = str(dtype_name)
        if self.imagery_type != "vhr":
            return "uint8_to_unit_interval"
        if dtype_name == "uint8":
            return "uint8_to_unit_interval"
        return "vhr_per_image_percentile_to_unit_interval"

    def _get_resolution(self, lucas_id, transform):
        csv_resolution = self.id_to_res.get(lucas_id)
        if csv_resolution is not None:
            return float(csv_resolution)

        try:
            if transform == Affine.identity():
                return getattr(self, "assumed_resolution", 1.0)
            return float(transform.a)
        except Exception:
            return getattr(self, "assumed_resolution", 1.0)

    def _get_crop_bounds(self, width, height, center_x, center_y, lucas_id, transform):
        resolution = self._get_resolution(lucas_id, transform)
        crop_width = int(self.patch_meters / resolution)
        crop_height = crop_width

        x0 = max(0, center_x - crop_width // 2)
        x1 = min(width, center_x + crop_width // 2)
        y0 = max(0, center_y - crop_height // 2)
        y1 = min(height, center_y + crop_height // 2)
        return x0, x1, y0, y1

    def _read_image_with_rasterio(self, path, lucas_id):
        return self._scale_to_unit_interval(self._read_raw_image_with_rasterio(path, lucas_id))

    def _read_image_with_opencv(self, path, lucas_id):
        return self._scale_to_unit_interval(self._read_raw_image_with_opencv(path, lucas_id))
