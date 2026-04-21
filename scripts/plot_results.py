import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import numpy as np
from collections import defaultdict
import argparse
from dataset import LucasDataset
import rasterio
import cv2
import geopandas as gpd
from shapely.geometry import Point


_desired_font = "Century Gothic"
_available_fonts = {f.name for f in fm.fontManager.ttflist}
if _desired_font in _available_fonts:
    plt.rcParams["font.family"] = _desired_font
else:
    print(f"Warning: '{_desired_font}' not found. Falling back to default sans-serif.")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
# Optional consistent styling
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["legend.frameon"] = False


def create_geopackage_with_predictions(csv_path, predictions, output_path):
    """
    Create a GeoPackage file with point locations and predicted labels.
    
    Args:
        csv_path: Path to the database CSV with geographic coordinates
        predictions: pandas.DataFrame or path to predictions CSV
        output_path: Path where to save the GeoPackage file
    """
    print(f"Creating GeoPackage with predictions...")
    
    # Load the database CSV with coordinates
    try:
        db_df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            db_df = pd.read_csv(csv_path, encoding='iso-8859-1')
        except UnicodeDecodeError:
            db_df = pd.read_csv(csv_path, encoding='windows-1252')
    
    # Load predictions (accept a DataFrame or a path)
    if isinstance(predictions, pd.DataFrame):
        pred_df = predictions.copy()
    else:
        pred_df = pd.read_csv(predictions)
    
    # Newer prediction CSVs provide top1_label/top1_prob. Use these directly (don't copy to other names).
    # Keep this block minimal: downstream logic will expect 'top1_label' / 'top1_prob'.
    
    # Convert IDPOINT/image_id and label columns to consistent types for merging and comparisons
    db_df['IDPOINT'] = db_df['IDPOINT'].astype(str)
    pred_df['image_id'] = pred_df['image_id'].astype(str)
    if 'true_label' in pred_df.columns:
        try:
            pred_df['true_label'] = pred_df['true_label'].astype(int)
        except Exception:
            pred_df['true_label'] = pred_df['true_label'].astype(float).astype(int)
    # Prefer top1_label as the canonical prediction column; fall back to predicted_label if present
    if 'top1_label' in pred_df.columns:
        try:
            pred_df['top1_label'] = pred_df['top1_label'].astype(int)
        except Exception:
            pred_df['top1_label'] = pred_df['top1_label'].astype(float).astype(int)
    elif 'predicted_label' in pred_df.columns:
        try:
            pred_df['predicted_label'] = pred_df['predicted_label'].astype(int)
        except Exception:
            pred_df['predicted_label'] = pred_df['predicted_label'].astype(float).astype(int)
    
    # Merge database with predictions on IDPOINT/image_id
    merged_df = db_df.merge(pred_df, left_on='IDPOINT', right_on='image_id', how='inner')
    
    # Check if we have coordinate columns
    coord_cols = ['LON', 'LAT', 'X_LAEA', 'Y_LAEA']
    missing_cols = [col for col in coord_cols if col not in merged_df.columns]
    if missing_cols:
        print(f"Warning: Missing coordinate columns: {missing_cols}")
        print(f"Available columns: {list(merged_df.columns)}")
    
    # Create geometry from LON/LAT coordinates (WGS84) or LAEA (preferred)
    if 'X_LAEA' in merged_df.columns and 'Y_LAEA' in merged_df.columns:
        # Use LAEA coordinates if available (these represent image centers)
        merged_df = merged_df.dropna(subset=['X_LAEA', 'Y_LAEA'])
        geometry = [Point(xy) for xy in zip(merged_df['X_LAEA'], merged_df['Y_LAEA'])]
        crs = 'EPSG:3035'  # LAEA Europe
    elif 'LON' in merged_df.columns and 'LAT' in merged_df.columns:
        merged_df = merged_df.dropna(subset=['LON', 'LAT'])
        geometry = [Point(xy) for xy in zip(merged_df['LON'], merged_df['LAT'])]
        crs = 'EPSG:4326'  # WGS84
    else:
        raise ValueError("No suitable coordinate columns found. Need either X_LAEA/Y_LAEA or LON/LAT")
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(merged_df, geometry=geometry, crs=crs)
    
    # Add class descriptions for better visualization in QGIS
    class_descriptions = {
        0: "Arable land",
        1: "Permanent crops",
        2: "Grassland",
        3: "Wooded Areas",
        4: "Shrubs",
        5: "Bare surface low or rare vegetation",
        6: "Artificial constructions and sealed areas",
        7: "Inland waters & Transitional water and coastal waters"
    }
    
    # Add readable class names (guard against missing columns)
    if 'true_label' in gdf.columns:
        gdf['true_class_name'] = gdf['true_label'].map(class_descriptions)
    else:
        gdf['true_class_name'] = None
    # Map predicted/top1 label to readable class name
    if 'top1_label' in gdf.columns:
        gdf['predicted_class_name'] = gdf['top1_label'].map(class_descriptions)
    elif 'predicted_label' in gdf.columns:
        gdf['predicted_class_name'] = gdf['predicted_label'].map(class_descriptions)
    else:
        gdf['predicted_class_name'] = None
    
    # Ensure we compute correctness using the top1 predicted label
    # Compute correctness using top1_label if available, otherwise predicted_label
    if 'true_label' in gdf.columns and 'top1_label' in gdf.columns:
        gdf['prediction_correct'] = (gdf['true_label'] == gdf['top1_label']).astype(int)
    elif 'true_label' in gdf.columns and 'predicted_label' in gdf.columns:
        gdf['prediction_correct'] = (gdf['true_label'] == gdf['predicted_label']).astype(int)
    else:
        gdf['prediction_correct'] = 0
    
    # Keep all prediction-related columns (top1/top2/probs/global_index) if present
    # Reorder columns to put important fields first
    # Use top1_label/top1_prob as canonical prediction/probability fields
    important_cols = ['IDPOINT', 'COUNTRY', 'true_label', 'top1_label', 
                     'true_class_name', 'predicted_class_name', 'prediction_correct',
                     'top1_prob', 'top1_probability', 'top2_label', 'top2_prob', 'global_index']
    
    # Get remaining columns (excluding geometry which is handled separately)
    remaining_cols = [col for col in gdf.columns if col not in important_cols + ['geometry']]
    final_col_order = [c for c in important_cols if c in gdf.columns] + remaining_cols + ['geometry']
    
    # Reorder columns
    gdf = gdf[final_col_order]
    
    # If GeoDataFrame CRS is not EPSG:3035 and we prefer to save in 3035, convert
    try:
        if gdf.crs is None:
            gdf.set_crs(crs, inplace=True)
        if gdf.crs.to_string() != 'EPSG:3035':
            gdf = gdf.to_crs('EPSG:3035')
    except Exception:
        # ignore CRS conversion errors but continue to save
        pass
    
    # Save to GeoPackage
    gdf.to_file(output_path, driver='GPKG')
    
    # Print accuracy summary
    accuracy = (gdf['prediction_correct'].sum() / len(gdf)) * 100 if len(gdf) > 0 else 0.0
    print(f"Overall accuracy: {accuracy:.1f}%")
    
    # Print per-country summary if COUNTRY column exists
    if 'COUNTRY' in gdf.columns:
        country_summary = gdf.groupby('COUNTRY').agg({
            'prediction_correct': ['count', 'sum'],
            'IDPOINT': 'count'
        }).round(1)
        country_summary.columns = ['Total', 'Correct', 'Count_Check']
        country_summary['Accuracy_%'] = (country_summary['Correct'] / country_summary['Total'] * 100).round(1)
        print(f"\nPer-country accuracy:")
        print(country_summary[['Total', 'Correct', 'Accuracy_%']])
    
    return gdf

def create_class_legend(dataset):
    """Create a legend mapping class numbers to their meanings."""
    # Standard LUCAS land cover classes (you can customize this)
    class_descriptions = {
        1: "Arable land",
        2: "Permanent crops",
        3: "Grassland",
        4: "Wooded Areas", 
        5: "Shrubs",
        6: "Bare surface low or rare vegetation",
        7: "Artificial constructions and sealed areas",
        8: "Inland waters & Transitional water and coastal waters"
    }
    

    reverse_mapping = {v: k for k, v in dataset.label_mapping.items()}
    
    # Filter out class 0 and create formatted legend with all text centered
    legend_lines = []
    legend_lines.append("Class Legend".center(25))
    legend_lines.append("─" * 25)
    
    for mapped_idx, original_class in sorted(reverse_mapping.items()):
        if original_class != 0:  # Exclude class 0
            description = class_descriptions.get(original_class, "Unknown")
            # Center each line
            line = f"Class {original_class}: {description}"
            legend_lines.append(line.center(25))
    
    return "\n".join(legend_lines)

def load_full_image(image_path):
    """Load the complete image without any cropping and return image with pixel resolution."""
    try:
        if image_path.endswith('.tif') or image_path.endswith('.tiff') or image_path.endswith('.png'):
            with rasterio.open(image_path) as src:
                # Get pixel resolution from transform
                pixel_resolution = abs(src.transform.a)  # meters per pixel
                
                # Read all bands or just RGB
                if src.count >= 3:
                    image = src.read([1, 2, 3])  # RGB bands
                    image = np.transpose(image, (1, 2, 0))  # Convert to HWC
                else:
                    image = src.read(1)  # Single band
                
                return image, pixel_resolution
        else:
            # Handle JPG and other formats with OpenCV
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Default to 0.3m/pixel for JPG images (common for high-res orthos)
            return image, 0.3
        
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None, None
            
    except Exception as e:
        print(f"Error loading full image {image_path}: {e}")
        return None, None


def process_image_for_display(image):
    """
    Simple image processing - just handle dimensions and basic display format.
    """
    # Convert tensor to numpy if needed
    if hasattr(image, 'numpy'):
        if image.shape[0] <= 3:  # CHW format (typical for PyTorch)
            image_np = image.permute(1, 2, 0).numpy()
        else:  # Already HWC
            image_np = image.numpy()
    else:
        image_np = image
    
    # Handle different channel configurations
    if len(image_np.shape) == 3:
        if image_np.shape[2] >= 3:
            # Use RGB channels
            display_img = image_np[:, :, :3]
        elif image_np.shape[2] == 1:
            # Single channel to grayscale
            display_img = image_np[:, :, 0]
        else:
            # Fallback
            display_img = image_np[:, :, 0]
    else:
        # Already 2D
        display_img = image_np
    
    return display_img


def draw_patch_overlay(ax, img_width, img_height, patch_meters=384):
    """Draw a rectangle showing the center patch area used by the algorithm"""
    # Get dataset info to calculate patch size in pixels
    # Assuming standard Sentinel-2 resolution of 10m per pixel
    pixel_resolution = 10  # meters per pixel
    patch_pixels = patch_meters / pixel_resolution
    
    # Calculate center patch position (use image coordinates)
    center_x = img_width / 2
    center_y = img_height / 2
    patch_half = patch_pixels / 2
    
    # Create rectangle patch with proper coordinates
    # Note: matplotlib imshow uses (x, y) where x is column and y is row
    rect = patches.Rectangle(
        (center_x - patch_half, center_y - patch_half),
        patch_pixels, patch_pixels,
        linewidth=1,  # Thinner border lines
        edgecolor='yellow',
        facecolor='none',
        linestyle='-',
        alpha=1.0
    )
    
    # Add rectangle to plot
    ax.add_patch(rect)
    """Draw a rectangle showing the center patch area used by the algorithm"""
    # Get dataset info to calculate patch size in pixels
    # Assuming standard Sentinel-2 resolution of 10m per pixel
    pixel_resolution = 10  # meters per pixel
    patch_pixels = patch_meters / pixel_resolution
    
    # Calculate center patch position (use image coordinates)
    center_x = img_width / 2
    center_y = img_height / 2
    patch_half = patch_pixels / 2
    
    # Create rectangle patch with proper coordinates
    # Note: matplotlib imshow uses (x, y) where x is column and y is row
    rect = patches.Rectangle(
        (center_x - patch_half, center_y - patch_half),
        patch_pixels, patch_pixels,
        linewidth=1,  # Thinner border lines
        edgecolor='yellow',
        facecolor='none',
        linestyle='-',
        alpha=1.0
    )
    
    # Add rectangle to plot
    ax.add_patch(rect)


def extract_center_patch(image, patch_ratio=0.15, patch_meters=None, pixel_resolution=10):
    """Extract a center patch as a ratio of the image size or fixed meter size."""
    if len(image.shape) == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = 1
    
    # If patch_meters is specified, calculate patch size in pixels
    if patch_meters is not None:
        patch_pixels = int(patch_meters / pixel_resolution)
        patch_h = patch_w = patch_pixels
    else:
        # Calculate patch size as ratio of image dimensions
        patch_h = int(h * patch_ratio)
        patch_w = int(w * patch_ratio)
    
    # Calculate center coordinates
    center_x, center_y = w // 2, h // 2
    half_patch_w = patch_w // 2
    half_patch_h = patch_h // 2
    
    # Extract patch boundaries
    x0 = max(0, center_x - half_patch_w)
    x1 = min(w, center_x + half_patch_w)
    y0 = max(0, center_y - half_patch_h)
    y1 = min(h, center_y + half_patch_h)
    
    # Extract patch
    if len(image.shape) == 3:
        patch = image[y0:y1, x0:x1, :]
    else:
        patch = image[y0:y1, x0:x1]
    
    return patch, (x0, y0, x1, y1)  # Return patch and coordinates


def draw_center_region_overlay(ax, img_width, img_height, patch_ratio=0.15, patch_meters=None, pixel_resolution=10):
    """Draw a yellow rectangle showing the center region being displayed as patch."""
    # If patch_meters is specified, calculate patch size in pixels
    if patch_meters is not None:
        patch_pixels = int(patch_meters / pixel_resolution)
        patch_w = patch_h = patch_pixels
    else:
        # Calculate patch dimensions as ratio
        patch_w = int(img_width * patch_ratio)
        patch_h = int(img_height * patch_ratio)
    
    # Calculate center coordinates
    center_x, center_y = img_width // 2, img_height // 2
    half_patch_w = patch_w // 2
    half_patch_h = patch_h // 2
    
    # Create rectangle patch
    rect = patches.Rectangle(
        (center_x - half_patch_w, center_y - half_patch_h),
        patch_w, patch_h,
        linewidth=2,
        edgecolor='yellow',
        facecolor='none',
        linestyle='-',
        alpha=0.8
    )
    
    # Add rectangle to plot
    ax.add_patch(rect)


def draw_center_point(ax, img_width, img_height, point_diameter_meters=20, pixel_resolution=10, add_compass=False):
    """Draw two circles showing the center points - 20m (blue) and 3m (red) diameter."""
    # Calculate center coordinates
    center_x = img_width / 2
    center_y = img_height / 2
    
    # Calculate radius for 20m circle
    large_radius_pixels = (20 / 2) / pixel_resolution
    
    # Calculate radius for 3m circle  
    small_radius_pixels = (3 / 2) / pixel_resolution
    
    # Create large circle (20m diameter) - Blue
    large_circle = patches.Circle(
        (center_x, center_y),
        large_radius_pixels,
        color='blue',
        alpha=0.8,
        linewidth=2,
        edgecolor='blue',
        facecolor='none',
        fill=False
    )
    
    # Create small circle (3m diameter) - Red
    small_circle = patches.Circle(
        (center_x, center_y),
        small_radius_pixels,
        color='red',
        alpha=0.9,
        linewidth=2,
        edgecolor='red',
        facecolor='none',
        fill=False
    )
    
    # Add circles to plot (large first, then small on top)
    ax.add_patch(large_circle)
    ax.add_patch(small_circle)
    
    # Add compass directions only if requested (for right plot)
    if add_compass:
        # North label - positioned slightly outside the blue circle
        north_x = center_x
        north_y = center_y - large_radius_pixels * 1.2  # Position outside blue circle
        ax.text(north_x, north_y, 'N', ha='center', va='center', fontsize=12, fontweight='bold', color='black',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', alpha=0.9))
        
        # East label - positioned slightly outside the blue circle  
        east_x = center_x + large_radius_pixels * 1.2  # Position outside blue circle
        east_y = center_y
        ax.text(east_x, east_y, 'E', ha='center', va='center', fontsize=12, fontweight='bold', color='black',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', alpha=0.9))


def plot_class_detailed_results(csv_path, image_dir, predictions_csv, output_dir, samples_per_class=3):
    """
    Plot detailed results for each class with full image and center patch side by side.
    Creates separate folders for each class.
    """
    print(f"Loading predictions from: {predictions_csv}")
    
    # Load predictions
    pred_df = pd.read_csv(predictions_csv)
    print(f"Loaded {len(pred_df)} predictions")
    # Prefer the new CSV columns 'top1_label' and 'top1_prob'
    if 'top1_label' in pred_df.columns:
        try:
            pred_df['top1_label'] = pred_df['top1_label'].astype(int)
        except Exception:
            pred_df['top1_label'] = pred_df['top1_label'].astype(float).astype(int)
    
    # Load dataset to access images and label mapping
    dataset = LucasDataset(image_dir, csv_path)
    print(f"Loaded dataset with {len(dataset)} images")
    
    # Load country information from database_corrected.csv
    db_df = pd.read_csv(csv_path)
    country_mapping = dict(zip(db_df['IDPOINT'].astype(str), db_df['COUNTRY']))
    
    # Country code to country name mapping
    country_codes = {
        'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CY': 'Cyprus', 'CZ': 'Czech Republic',
        'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia', 'ES': 'Spain', 'FI': 'Finland',
        'FR': 'France', 'GR': 'Greece', 'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland',
        'IT': 'Italy', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MT': 'Malta',
        'NL': 'Netherlands', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'SE': 'Sweden',
        'SI': 'Slovenia', 'SK': 'Slovakia'
    }
    
    # Create class descriptions
    class_descriptions = {
        1: "Arable land",
        2: "Permanent crops",
        3: "Grassland",
        4: "Wooded Areas", 
        5: "Shrubs",
        6: "Bare surface low or rare vegetation",
        7: "Artificial constructions and sealed areas",
        8: "Inland waters & Transitional water and coastal waters"
    }
    
    
    # Create mapping from image_id to dataset index (use string keys)
    id_to_idx = {}
    for idx, (image_path, lucas_id) in enumerate(dataset.image_files):
        id_to_idx[str(lucas_id)] = idx

    # Build mapping from IDPOINT to spatial_resolution from the master CSV (use CSV as authoritative source)
    id_to_res = {}
    if 'spatial_resolution' in db_df.columns:
        for _, r in db_df[['IDPOINT', 'spatial_resolution']].dropna(subset=['IDPOINT']).iterrows():
            try:
                id_str = str(int(float(r['IDPOINT']))) if not isinstance(r['IDPOINT'], str) else r['IDPOINT']
            except Exception:
                id_str = str(r['IDPOINT'])
            try:
                res = float(r['spatial_resolution'])
            except Exception:
                res = None
            if res is not None:
                id_to_res[id_str] = res
    else:
        print("Warning: 'spatial_resolution' column not found in master CSV. Per-user policy, plotting requires this column.")
    

    
    # Group predictions by class and correctness
    correct_samples = defaultdict(list)
    incorrect_samples = defaultdict(list)
    
    reverse_mapping = {v: k for k, v in dataset.label_mapping.items()}
    
    # Check if confidence scores are available (prefer top1_prob/top1_probability)
    has_confidence = ('top1_prob' in pred_df.columns) or ('top1_probability' in pred_df.columns) or ('confidence' in pred_df.columns) or ('probability' in pred_df.columns)
    confidence_col = (
        'top1_prob' if 'top1_prob' in pred_df.columns else
        ('top1_probability' if 'top1_probability' in pred_df.columns else
         ('confidence' if 'confidence' in pred_df.columns else
          ('probability' if 'probability' in pred_df.columns else None))))
    

    
    # Process predictions
    matched_count = 0
    for _, row in pred_df.iterrows():
        # Convert float ID to integer string (remove .0)
        img_id = str(int(float(row['image_id'])))
        true_label = int(row['true_label'])
        # Read prediction from top1_label if available, otherwise fall back to predicted_label
        if 'top1_label' in pred_df.columns:
            pred_label = int(row['top1_label'])
        else:
            pred_label = int(row['predicted_label'])

        # Get confidence score if available
        confidence = float(row[confidence_col]) if has_confidence and confidence_col in row.index and not pd.isna(row[confidence_col]) else None
        
        if img_id not in id_to_idx:
            continue
            
        matched_count += 1
        dataset_idx = id_to_idx[img_id]
        original_class = reverse_mapping.get(true_label, true_label)
        
        if pred_label == true_label:
            correct_samples[original_class].append((dataset_idx, img_id, pred_label, true_label, confidence))
        else:
            incorrect_samples[original_class].append((dataset_idx, img_id, pred_label, true_label, confidence))
    
    print(f"Successfully matched {matched_count} out of {len(pred_df)} predictions with dataset images")
    
    # Get unique classes present in the data
    all_classes = set(reverse_mapping.get(int(row['true_label']), int(row['true_label'])) for _, row in pred_df.iterrows())
    unique_classes = sorted(all_classes)
    

    
    # Create output directories
    for class_label in unique_classes:
        class_dir = os.path.join(output_dir, f"class_{class_label}")
        os.makedirs(class_dir, exist_ok=True)
    
    # Process each class
    for class_label in unique_classes:
        class_description = class_descriptions.get(class_label, "Unknown")
        class_dir = os.path.join(output_dir, f"class_{class_label}")
        
        print(f"Processing Class {class_label}: {class_description}")
        
                # Plot correct predictions
        if correct_samples[class_label]:
            for idx, sample_data in enumerate(correct_samples[class_label][:samples_per_class]):
                # Handle both old format (4 items) and new format (5 items with confidence)
                if len(sample_data) == 5:
                    dataset_idx, img_id, pred, target, confidence = sample_data
                else:
                    dataset_idx, img_id, pred, target = sample_data
                    confidence = None
                try:
                    # Get the full image path from dataset
                    image_path, _ = dataset.image_files[dataset_idx]
                    
                    # Load the complete image (we will prefer CSV-provided spatial_resolution but fall back to raster)
                    full_image, img_pixel_res = load_full_image(image_path)
                    if full_image is None:
                        continue

                    # Prefer spatial_resolution from master CSV (id_to_res); fall back to raster-reported resolution
                    pixel_resolution = id_to_res.get(str(img_id), img_pixel_res)
                    # If still missing or zero, choose sensible defaults (10 m/px for Sentinel-like, 0.3 for JPG if raster gave none)
                    if pixel_resolution is None or pixel_resolution == 0:
                        fallback = img_pixel_res if img_pixel_res is not None and img_pixel_res > 0 else 10
                        print(f"  Warning: spatial_resolution missing for {img_id} in master CSV; falling back to {fallback} m/px")
                        pixel_resolution = fallback

                    # Extract center patch (384 meters) using the determined pixel resolution
                    center_patch, patch_coords = extract_center_patch(full_image, patch_meters=384, pixel_resolution=pixel_resolution)
                    
                    # Process images for display
                    full_img_disp = process_image_for_display(full_image)
                    patch_img_disp = process_image_for_display(center_patch)
                    
                    # Create figure with 1 row, 2 columns
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Plot full image with yellow overlay showing the patch region
                    ax1.imshow(full_img_disp, cmap='gray' if len(full_img_disp.shape) == 2 else None)
                    draw_center_point(ax1, full_image.shape[1], full_image.shape[0], pixel_resolution=pixel_resolution)
                    draw_center_region_overlay(ax1, full_image.shape[1], full_image.shape[0], patch_meters=384, pixel_resolution=pixel_resolution)
                    draw_center_point(ax1, full_image.shape[1], full_image.shape[0], pixel_resolution=pixel_resolution)
                    ax1.set_title(f"Full Image\nClass {class_label}: {class_description}\nLucas ID: {img_id}", fontsize=12)
                    ax1.axis('off')
                    
                    # Add country name to bottom left of first image
                    country_code = country_mapping.get(str(img_id), 'Unknown')
                    country_name = country_codes.get(country_code, country_code)
                    ax1.text(0.02, 0.02, f"Country: {country_name}", transform=ax1.transAxes, 
                            fontsize=10, color='white', fontweight='light',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
                    
                    # Plot center patch
                    ax2.imshow(patch_img_disp, cmap='gray' if len(patch_img_disp.shape) == 2 else None)
                    # Draw circles on the zoomed patch for better visibility (use same pixel resolution) and add compass
                    draw_center_point(ax2, center_patch.shape[1], center_patch.shape[0], pixel_resolution=pixel_resolution, add_compass=True)
                    ax2.set_title(f"Center Region (384m x 384m)\nUsed for Classification", fontsize=12)
                    ax2.axis('off')                    # Add legend to the second subplot
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        patches.Patch(edgecolor='yellow', facecolor='none', label='Patch region (area used for classification)', linewidth=2),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, alpha=0.8, label='Blue Circle: EWO (20m)'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, alpha=0.9, label='Red Circle: Homogeneous Plot (3m)')
                    ]
                    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
                    
                    # Add confidence score to bottom right of the zoomed image if available
                    if confidence is not None:
                        ax2.text(0.98, 0.02, f"Confidence: {confidence:.3f}", transform=ax2.transAxes, 
                                fontsize=9, color='white', fontweight='light', ha='right', va='bottom',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='darkgreen', alpha=0.8))
                        
                    if pixel_resolution is not None: 
                            ax2.text(0.98, 0.08, f"Spatial Res: {pixel_resolution:.1f} m/px", transform=ax2.transAxes,
                                    fontsize=9, color='white', fontweight='light', ha='right', va='bottom',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgrey', alpha=0.8))
                    
                    # Add overall title (without confidence since it's now on the image)
                    title = f"CORRECT Prediction - Class {class_label}: {class_description}"
                    
                    fig.suptitle(title, fontsize=12, fontweight='bold', color='darkgreen', fontfamily='serif')
                    
                    # Save the plot
                    filename = f"class_{class_label}_correct_{idx+1}_ID_{img_id}.png"
                    filepath = os.path.join(class_dir, filename)
                    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave smaller space for title
                    plt.savefig(filepath, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    print(f"  Error processing correct prediction {img_id}: {e}")
                    continue
        
        # Plot incorrect predictions
        if incorrect_samples[class_label]:
            for idx, sample_data in enumerate(incorrect_samples[class_label][:samples_per_class]):
                # Handle both old format (4 items) and new format (5 items with confidence)
                if len(sample_data) == 5:
                    dataset_idx, img_id, pred, target, confidence = sample_data
                else:
                    dataset_idx, img_id, pred, target = sample_data
                    confidence = None
                try:
                    # Get the full image path from dataset
                    image_path, _ = dataset.image_files[dataset_idx]
                    
                    # Load the complete image (prefer CSV-provided spatial_resolution but fall back to raster)
                    full_image, img_pixel_res = load_full_image(image_path)
                    if full_image is None:
                        continue

                    # Prefer spatial_resolution from master CSV (id_to_res); fall back to raster-reported resolution
                    pixel_resolution = id_to_res.get(str(img_id), img_pixel_res)
                    if pixel_resolution is None or pixel_resolution == 0:
                        fallback = img_pixel_res if img_pixel_res is not None and img_pixel_res > 0 else 10
                        print(f"  Warning: spatial_resolution missing for {img_id} in master CSV; falling back to {fallback} m/px")
                        pixel_resolution = fallback

                    # Extract center patch (384 meters) using the determined pixel resolution
                    center_patch, patch_coords = extract_center_patch(full_image, patch_meters=384, pixel_resolution=pixel_resolution)
                    
                    # Process images for display
                    full_img_disp = process_image_for_display(full_image)
                    patch_img_disp = process_image_for_display(center_patch)
                    
                    # Create figure with 1 row, 2 columns
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Plot full image with yellow overlay showing the patch region
                    ax1.imshow(full_img_disp, cmap='gray' if len(full_img_disp.shape) == 2 else None)
                    draw_center_region_overlay(ax1, full_image.shape[1], full_image.shape[0], patch_meters=384, pixel_resolution=pixel_resolution)
                    draw_center_point(ax1, full_image.shape[1], full_image.shape[0], pixel_resolution=pixel_resolution)
                    ax1.set_title(f"Full Image\nClass {class_label}: {class_description}\nLucas ID: {img_id}", fontsize=12)
                    ax1.axis('off')
                    
                    # Add country name to bottom left of first image
                    country_code = country_mapping.get(str(img_id), 'Unknown')
                    country_name = country_codes.get(country_code, country_code)
                    ax1.text(0.02, 0.02, f"Country: {country_name}", transform=ax1.transAxes, 
                            fontsize=10, color='white', fontweight='light',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
                    
                    # Plot center patch
                    ax2.imshow(patch_img_disp, cmap='gray' if len(patch_img_disp.shape) == 2 else None)
                    # Draw circles on the zoomed patch for better visibility (use same pixel resolution) and add compass
                    draw_center_point(ax2, center_patch.shape[1], center_patch.shape[0], pixel_resolution=pixel_resolution, add_compass=True)
                    ax2.set_title(f"Center Region (384m x 384m)\nUsed for Classification", fontsize=12)
                    ax2.axis('off')
                    
                    # Add legend to the second subplot
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        patches.Patch(edgecolor='yellow', facecolor='none', label='Patch region (area used for classification)', linewidth=2),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, alpha=0.8, label='Blue Circle: EWO (20m)'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, alpha=0.9, label='Red Circle: Homogeneous Plot (3m)')
                    ]
                    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
                    
                    # Add confidence score to bottom right of the zoomed image if available
                    if confidence is not None:
                        ax2.text(0.98, 0.02, f"Confidence: {confidence:.3f}", transform=ax2.transAxes, 
                                fontsize=9, color='white', fontweight='light', ha='right', va='bottom',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='darkred', alpha=0.8))
                    
                    if pixel_resolution is not None:
                            ax2.text(0.98, 0.08, f"Spatial Res: {pixel_resolution:.1f} m/px", transform=ax2.transAxes,
                                    fontsize=9, color='white', fontweight='light', ha='right', va='bottom',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgrey', alpha=0.8))
                    
                    # Add overall title with prediction info (without confidence since it's now on the image)
                    pred_class_desc = class_descriptions.get(reverse_mapping.get(pred, pred), "Unknown")
                    title = f"INCORRECT - True: Class {class_label} ({class_description}), Predicted: Class {reverse_mapping.get(pred, pred)} ({pred_class_desc})"
                    
                    fig.suptitle(title, fontsize=12, fontweight='bold', color='darkred', fontfamily='serif')
                    
                    # Save the plot
                    filename = f"class_{class_label}_incorrect_{idx+1}_ID_{img_id}_pred_{reverse_mapping.get(pred, pred)}.png"
                    filepath = os.path.join(class_dir, filename)
                    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave smaller space for title
                    plt.savefig(filepath, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    print(f"  Error processing incorrect prediction {img_id}: {e}")
                    continue
        
        # Create summary file for this class
        summary_path = os.path.join(class_dir, f"class_{class_label}_summary.txt")
        with open(summary_path, 'w') as f:
            n_correct = len(correct_samples[class_label])
            n_incorrect = len(incorrect_samples[class_label])
            total = n_correct + n_incorrect
            accuracy = n_correct / total if total > 0 else 0
            
            f.write(f"Class {class_label}: {class_description}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total samples: {total}\n")
            f.write(f"Correct predictions: {n_correct}\n")
            f.write(f"Incorrect predictions: {n_incorrect}\n")
            f.write(f"Accuracy: {accuracy:.1%}\n")
            f.write(f"\nVisualization Details:\n")
            f.write(f"Center region\n")
            f.write(f"Region shown: Center area with detailed zoom for better visibility\n")
            f.write(f"Yellow overlay: Shows the exact region used for classification\n")
            f.write(f"Blue circle: EWO (20m diameter) - Environmental reference\n")
            f.write(f"Red circle: Homogeneous Plot (3m diameter) - Precise measurement location\n")
            f.write(f"Note: Circle sizes are dynamically calculated based on actual image pixel resolution\n")
        
        print(f"  Created summary: class_{class_label} ({class_description}) - {len(correct_samples[class_label])} correct, {len(incorrect_samples[class_label])} incorrect")
    



def plot_classification_results(csv_path, image_dir, predictions_csv, output_dir, samples_per_class=3):
    """
    Plot sample images for each class showing correct and incorrect predictions.
    """
    print(f"Loading predictions from: {predictions_csv}")
    
    # Load predictions
    pred_df = pd.read_csv(predictions_csv)
    print(f"Loaded {len(pred_df)} predictions")
    
    # Load dataset to access images and label mapping
    dataset = LucasDataset(image_dir, csv_path)
    print(f"Loaded dataset with {len(dataset)} images")
    
    # Create mapping from image_id to dataset index
    id_to_idx = {}
    for idx, (image_path, lucas_id) in enumerate(dataset.image_files):
        id_to_idx[lucas_id] = idx
    
    # Group predictions by class and correctness
    correct_samples = defaultdict(list)
    incorrect_samples = defaultdict(list)
    
    reverse_mapping = {v: k for k, v in dataset.label_mapping.items()}
    
    # Process predictions
    for _, row in pred_df.iterrows():
        img_id = str(row['image_id'])
        true_label = int(row['true_label'])
        # Prefer top1_label if available
        if 'top1_label' in pred_df.columns:
            pred_label = int(row['top1_label'])
        else:
            pred_label = int(row['predicted_label'])
        
        if img_id not in id_to_idx:
            continue
            
        dataset_idx = id_to_idx[img_id]
        original_class = reverse_mapping.get(true_label, true_label)
        
        if pred_label == true_label:
            correct_samples[original_class].append((dataset_idx, img_id, pred_label, true_label))
        else:
            incorrect_samples[original_class].append((dataset_idx, img_id, pred_label, true_label))
    
    # Get unique classes present in the data
    all_classes = set(reverse_mapping.get(int(row['true_label']), int(row['true_label'])) for _, row in pred_df.iterrows())
    unique_classes = sorted(all_classes)
    
    # Calculate grid size
    n_classes = len(unique_classes)
    cols = min(6, n_classes)  # Max 6 columns
    rows = (n_classes + cols - 1) // cols
    
    # Create class legend
    legend_text = create_class_legend(dataset)
    
    # Create figure for correct predictions
    fig_correct, axes_correct = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes_correct = [axes_correct] if cols == 1 else axes_correct
    else:
        axes_correct = axes_correct.flatten()
    
    # Create figure for incorrect predictions
    fig_incorrect, axes_incorrect = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes_incorrect = [axes_incorrect] if cols == 1 else axes_incorrect
    else:
        axes_incorrect = axes_incorrect.flatten()
    
    for class_idx, class_label in enumerate(unique_classes):
        # Plot correct prediction sample
        ax_correct = axes_correct[class_idx]
        if correct_samples[class_label]:
            # Try multiple samples to find a good one
            for sample_idx in range(min(3, len(correct_samples[class_label]))):
                dataset_idx, img_id, pred, target = correct_samples[class_label][sample_idx]
                # Get the full image path from dataset
                image_path, _ = dataset.image_files[dataset_idx]
                
                # Load the complete image (not cropped)
                try:
                    full_image, _ = load_full_image(image_path)  # Ignore pixel resolution for grid view
                    if full_image is not None:
                        img_disp = process_image_for_display(full_image)
                        # Check if image is valid for display
                        if hasattr(img_disp, 'max') and img_disp.max() > img_disp.min():
                            break  # Good image found
                except Exception as e:
                    print(f"Error processing full image {img_id}: {e}")
                    continue
            else:
                # No good image found, show placeholder
                ax_correct.text(0.5, 0.5, f"Class {class_label}\nNo suitable\ncorrect image", 
                              ha='center', va='center', transform=ax_correct.transAxes)
                ax_correct.axis('off')
                continue
                
            ax_correct.imshow(img_disp, cmap='gray' if len(img_disp.shape) == 2 else None)
            ax_correct.set_title(f"Class {class_label}\n✓ Correct\nPred: {reverse_mapping.get(pred, pred)}", 
                               fontsize=10, color='green')
            # Add Lucas ID in small font at bottom left corner
            ax_correct.text(0.02, 0.02, f"ID: {img_id}", transform=ax_correct.transAxes, 
                          fontsize=8, color='white', bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            ax_correct.axis('off')
        else:
            ax_correct.text(0.5, 0.5, f"Class {class_label}\nNo correct\npredictions", 
                          ha='center', va='center', transform=ax_correct.transAxes)
            ax_correct.axis('off')
        
        # Plot incorrect prediction sample
        ax_incorrect = axes_incorrect[class_idx]
        if incorrect_samples[class_label]:
            # Try multiple samples to find a good one
            for sample_idx in range(min(3, len(incorrect_samples[class_label]))):
                dataset_idx, img_id, pred, target = incorrect_samples[class_label][sample_idx]
                # Get the full image path from dataset
                image_path, _ = dataset.image_files[dataset_idx]
                
                # Load the complete image (not cropped)
                try:
                    full_image, _ = load_full_image(image_path)  # Ignore pixel resolution for grid view
                    if full_image is not None:
                        img_disp = process_image_for_display(full_image)
                        # Check if image is valid for display
                        if hasattr(img_disp, 'max') and img_disp.max() > img_disp.min():
                            break  # Good image found
                except Exception as e:
                    print(f"Error processing full image {img_id}: {e}")
                    continue
            else:
                # No good image found, show placeholder
                ax_incorrect.text(0.5, 0.5, f"Class {class_label}\nNo suitable\nincorrect image", 
                                ha='center', va='center', transform=ax_incorrect.transAxes)
                ax_incorrect.axis('off')
                continue
                
            ax_incorrect.imshow(img_disp, cmap='gray' if len(img_disp.shape) == 2 else None)
            ax_incorrect.set_title(f"Class {class_label}\n✗ Incorrect\nTrue: {class_label}, Pred: {reverse_mapping.get(pred, pred)}", 
                                 fontsize=10, color='red')
            # Add Lucas ID in small font at bottom left corner
            ax_incorrect.text(0.02, 0.02, f"ID: {img_id}", transform=ax_incorrect.transAxes, 
                            fontsize=8, color='white', bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            ax_incorrect.axis('off')
        else:
            ax_incorrect.text(0.5, 0.5, f"Class {class_label}\nAll predictions\ncorrect!", 
                            ha='center', va='center', transform=ax_incorrect.transAxes, color='green')
            ax_incorrect.axis('off')
    
    # Hide empty subplots
    for i in range(len(unique_classes), len(axes_correct)):
        axes_correct[i].axis('off')
        axes_incorrect[i].axis('off')
    
    # Add legend to the plots
    fig_correct.text(0.98, 0.02, legend_text, transform=fig_correct.transFigure, 
                    fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    fig_incorrect.text(0.98, 0.02, legend_text, transform=fig_incorrect.transFigure, 
                      fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    # Save plots
    fig_correct.suptitle('Correct Predictions by Class', fontsize=16, fontweight='bold')
    fig_correct.tight_layout()
    correct_path = os.path.join(output_dir, "correct_predictions.png")
    fig_correct.savefig(correct_path, dpi=150, bbox_inches='tight')
    
    fig_incorrect.suptitle('Incorrect Predictions by Class', fontsize=16, fontweight='bold')
    fig_incorrect.tight_layout()
    incorrect_path = os.path.join(output_dir, "incorrect_predictions.png")
    fig_incorrect.savefig(incorrect_path, dpi=150, bbox_inches='tight')
    
    plt.close(fig_correct)
    plt.close(fig_incorrect)
    
    print(f"Saved correct predictions plot to: {correct_path}")
    print(f"Saved incorrect predictions plot to: {incorrect_path}")
    
    # Print summary statistics
    print(f"\n=== Classification Summary ===")
    for class_label in unique_classes:
        n_correct = len(correct_samples[class_label])
        n_incorrect = len(incorrect_samples[class_label])
        total = n_correct + n_incorrect
        accuracy = n_correct / total if total > 0 else 0
        print(f"Class {class_label}: {n_correct}/{total} correct ({accuracy:.1%})")


def load_image_ids_from_file(id_file_path):
    """
    Load image IDs from a text file.
    
    Args:
        id_file_path: Path to the text file containing image IDs (one per line)
    
    Returns:
        set: Set of image IDs as strings
    """
    if not os.path.exists(id_file_path):
        raise FileNotFoundError(f"Image ID file not found: {id_file_path}")
    
    with open(id_file_path, 'r') as f:
        image_ids = {line.strip() for line in f if line.strip()}
    
    print(f"Loaded {len(image_ids)} image IDs from {id_file_path}")
    return image_ids


def filter_predictions_by_ids(predictions_csv, image_ids, output_csv=None):
    """
    Filter predictions to only include specified image IDs.
    
    Args:
        predictions_csv: Path to original predictions CSV
        image_ids: Set of image IDs to include
        output_csv: Optional path to save filtered predictions
    
    Returns:
        pd.DataFrame: Filtered predictions dataframe
    """
    # Load all predictions
    pred_df = pd.read_csv(predictions_csv)
    
    # Convert image_id to string for consistent comparison
    pred_df['image_id'] = pred_df['image_id'].astype(str)
    
    # Filter to only include specified IDs
    filtered_df = pred_df[pred_df['image_id'].isin(image_ids)].copy()
    
    print(f"Filtered predictions from {len(pred_df)} to {len(filtered_df)} samples")
    
    if output_csv:
        filtered_df.to_csv(output_csv, index=False)
        print(f"Saved filtered predictions to: {output_csv}")
    
    return filtered_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot classification results from predictions")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the image directory")
    parser.add_argument("--predictions_csv", type=str, required=True, help="Path to predictions.csv file")
    parser.add_argument("--output_dir", type=str, default="detailed_plots", help="Output directory for plots")
    parser.add_argument("--mode", type=str, choices=["simple", "detailed"], default="detailed", 
                        help="Plotting mode: 'simple' for overview, 'detailed' for per-class folders")
    parser.add_argument("--samples_per_class", type=int, default=3, help="Number of samples per class to plot")
    parser.add_argument("--no_gpkg", action="store_true", default=False, help="Skip GeoPackage creation")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions (expect the provided CSV to already contain the test/validation rows you want)
    predictions_to_use = args.predictions_csv
    pred_df = pd.read_csv(args.predictions_csv)
    
    # Plot confusion matrix as well
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    y_true = pred_df['true_label']
    # Use top1_label as canonical prediction column if present
    if 'top1_label' in pred_df.columns:
        y_pred = pred_df['top1_label']
    else:
        y_pred = pred_df['predicted_label']
    # Get class names from dataset
    dataset = LucasDataset(args.image_dir, args.csv_path)
    class_names = [
        "Arable land",
        "Permanent crops",
        "Grassland",
        "Wooded Areas",
        "Shrubs",
        "Bare surface low or rare vegetation",
        "Artificial constructions and sealed areas",
        "Inland waters & Transitional water and coastal waters"
    ]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    # Set x-axis tick labels to italic and smaller font
    plt.xticks(fontsize=9, fontstyle='italic', rotation=45, ha='right')
    plt.yticks(fontsize=9)
    plt.tight_layout()
    cm_img_path = os.path.join(args.output_dir, 'predictions_confusion_matrix.png')
    plt.savefig(cm_img_path)
    print(f"Confusion matrix image saved to: {cm_img_path}")

    # Create GeoPackage with predictions
    if not args.no_gpkg:
        gpkg_path = os.path.join(args.output_dir, 'predictions_points.gpkg')
        try:
            # Pass the filtered/full predictions DataFrame to the geopackage creator so it uses the correct set
            gdf = create_geopackage_with_predictions(
                csv_path=args.csv_path,
                predictions=pred_df,
                output_path=gpkg_path
            )
        except Exception as e:
            print(f"Error creating GeoPackage: {e}")
            print("Make sure you have geopandas installed: pip install geopandas")
    else:
        print("Skipping GeoPackage creation (--no_gpkg flag was used)")

    if args.mode == "detailed":

        predictions_dir = os.path.dirname(predictions_to_use)
        detailed_output_dir = os.path.join(predictions_dir, "detailed_plots")
        
        plot_class_detailed_results(
            csv_path=args.csv_path,
            image_dir=args.image_dir,
            predictions_csv=predictions_to_use,
            output_dir=detailed_output_dir, 
            samples_per_class=args.samples_per_class
    )
    else:
        plot_classification_results(
            csv_path=args.csv_path,
            image_dir=args.image_dir,
            predictions_csv=predictions_to_use,
            output_dir=args.output_dir
        )
