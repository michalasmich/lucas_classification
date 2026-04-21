import os
import numpy as np
import pandas as pd
from PIL import Image
import argparse
from collections import defaultdict, Counter
import torch
from tqdm import tqdm
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not available. GeoTIFF files may not be readable.")

def get_image_info(image_path):
    """Get image dimensions and resolution info"""
    try:
        # Try to read with rasterio first (for GeoTIFF files)
        if HAS_RASTERIO and image_path.lower().endswith(('.tif', '.tiff')):
            try:
                with rasterio.open(image_path) as src:
                    width = src.width
                    height = src.height
                    # Get resolution from transform if available
                    if src.transform is not None:
                        resolution = abs(src.transform[0])  # pixel size in x direction
                        resolution = f"{resolution:.2f}"
                    else:
                        resolution = "Unknown"
                    return width, height, resolution
            except Exception:
                pass  # Fall back to PIL
        
        # Use PIL for other formats or as fallback
        with Image.open(image_path) as img:
            width, height = img.size
            # Try to get DPI info if available
            dpi = img.info.get('dpi', (None, None))
            if dpi[0] is not None:
                resolution = dpi[0]
            else:
                # Default resolution if not available
                resolution = "Unknown"
            
            return width, height, resolution
            
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None, None, None

def analyze_dataset(images_dir, csv_path):
    """Analyze the dataset structure and provide statistics"""
    
    if not csv_path:
        print("Error: CSV path is required to read class labels!")
        return
    
    print(f"Analyzing dataset in: {images_dir}")
    print(f"Reading labels from: {csv_path}")
    print("=" * 60)
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} records")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check if train/test subdirectories exist
    train_dir = os.path.join(images_dir, "train")
    test_dir = os.path.join(images_dir, "test")
    
    has_train_test_split = os.path.exists(train_dir) and os.path.exists(test_dir)
    
    if has_train_test_split:
        print("Found train/test split structure")
        directories = {"train": train_dir, "test": test_dir}
    else:
        print("No train/test split found, analyzing entire directory")
        directories = {"all": images_dir}
    
    # Create a mapping from LucasId to class (excluding class 10)
    excluded_classes = [10]
    df_filtered = df[~df['STR25'].isin(excluded_classes)]
    lucas_to_class = dict(zip(df_filtered['lucasId'].astype(str), df_filtered['STR25']))
    
    print(f"Found {len(lucas_to_class)} valid records (excluding class 10)")
    
    # Statistics containers
    total_stats = {
        "class_counts": defaultdict(int),
        "dimension_counts": defaultdict(int),
        "resolution_counts": defaultdict(int),
        "total_images": 0,
        "images_with_labels": 0,
        "images_without_labels": 0
    }
    
    split_stats = {}
    
    # Analyze each directory
    for split_name, dir_path in directories.items():
        print(f"\nAnalyzing {split_name} directory...")
        
        stats = {
            "class_counts": defaultdict(int),
            "dimension_counts": defaultdict(int),
            "resolution_counts": defaultdict(int),
            "total_images": 0,
            "images_with_labels": 0,
            "images_without_labels": 0
        }
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        image_files = []
        
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        print(f"Found {len(image_files)} images in {split_name}")
        
        # Analyze each image
        for image_path in tqdm(image_files, desc=f"Processing {split_name}"):
            try:
                # Extract Lucas ID from filename
                filename = os.path.basename(image_path)
                # Remove extension and extract ID
                name_without_ext = os.path.splitext(filename)[0]
                
                # Try different patterns to extract Lucas ID
                lucas_id = None
                if name_without_ext.startswith('ID-'):
                    lucas_id = name_without_ext[3:].split('_')[0]
                elif '_' in name_without_ext:
                    # Try first part before underscore
                    parts = name_without_ext.split('_')
                    if parts[0].isdigit():
                        lucas_id = parts[0]
                
                # Get class from CSV mapping
                if lucas_id and lucas_id in lucas_to_class:
                    class_id = lucas_to_class[lucas_id]
                    stats["images_with_labels"] += 1
                    total_stats["images_with_labels"] += 1
                else:
                    class_id = "No Label"
                    stats["images_without_labels"] += 1
                    total_stats["images_without_labels"] += 1
                
                # Get image info
                width, height, resolution = get_image_info(image_path)
                
                if width is not None and height is not None:
                    stats["class_counts"][class_id] += 1
                    stats["dimension_counts"][f"{width}x{height}"] += 1
                    stats["resolution_counts"][resolution] += 1
                    stats["total_images"] += 1
                    
                    # Add to total stats
                    total_stats["class_counts"][class_id] += 1
                    total_stats["dimension_counts"][f"{width}x{height}"] += 1
                    total_stats["resolution_counts"][resolution] += 1
                    total_stats["total_images"] += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        split_stats[split_name] = stats
    
    # Print results
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS RESULTS")
    print("=" * 60)
    
    # Overall statistics
    print(f"\nTOTAL IMAGES: {total_stats['total_images']:,}")
    print(f"Images with labels: {total_stats['images_with_labels']:,}")
    print(f"Images without labels: {total_stats['images_without_labels']:,}")
    
    # Class distribution
    print(f"\nCLASS DISTRIBUTION:")
    print("-" * 30)
    class_items = sorted(total_stats["class_counts"].items())
    for class_id, count in class_items:
        if class_id != "No Label":
            print(f"Class {class_id}: {count:,} images")
        else:
            print(f"No Label: {count:,} images")
    
    # Dimensions distribution
    print(f"\nDIMENSIONS DISTRIBUTION:")
    print("-" * 30)
    dim_items = sorted(total_stats["dimension_counts"].items(), 
                       key=lambda x: x[1], reverse=True)
    for dimensions, count in dim_items:
        print(f"Dimension {dimensions}: {count:,} images")
    
    # Resolution distribution
    print(f"\nRESOLUTION DISTRIBUTION:")
    print("-" * 30)
    res_items = sorted(total_stats["resolution_counts"].items(), 
                       key=lambda x: x[1], reverse=True)
    for resolution, count in res_items:
        print(f"Resolution {resolution}: {count:,} images")
    
    # Split-specific statistics
    if has_train_test_split:
        for split_name, stats in split_stats.items():
            print(f"\n{split_name.upper()} SPLIT DETAILS:")
            print("-" * 30)
            print(f"Total images: {stats['total_images']:,}")
            
            print(f"\nClass distribution in {split_name}:")
            class_items = sorted(stats["class_counts"].items())
            for class_id, count in class_items:
                percentage = (count / stats['total_images']) * 100 if stats['total_images'] > 0 else 0
                print(f"  Class {class_id}: {count:,} images ({percentage:.1f}%)")
            
            print(f"\nTop dimensions in {split_name}:")
            dim_items = sorted(stats["dimension_counts"].items(), 
                             key=lambda x: x[1], reverse=True)[:5]
            for dimensions, count in dim_items:
                percentage = (count / stats['total_images']) * 100 if stats['total_images'] > 0 else 0
                print(f"  {dimensions}: {count:,} images ({percentage:.1f}%)")
    
    # Summary statistics
    print(f"\nSUMMARY:")
    print("-" * 30)
    print(f"Total unique classes: {len(total_stats['class_counts'])}")
    print(f"Total unique dimensions: {len(total_stats['dimension_counts'])}")
    print(f"Total unique resolutions: {len(total_stats['resolution_counts'])}")
    
    if has_train_test_split:
        train_count = split_stats.get("train", {}).get("total_images", 0)
        test_count = split_stats.get("test", {}).get("total_images", 0)
        train_pct = (train_count / total_stats['total_images']) * 100 if total_stats['total_images'] > 0 else 0
        test_pct = (test_count / total_stats['total_images']) * 100 if total_stats['total_images'] > 0 else 0
        
        print(f"Train/Test split: {train_count:,} ({train_pct:.1f}%) / {test_count:,} ({test_pct:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Analyze image dataset structure and statistics")
    parser.add_argument("--images_dir", type=str, default="images4", 
                       help="Path to the images directory")
    parser.add_argument("--csv_path", type=str, default="STR25.csv",
                       help="Path to CSV file with labels")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.images_dir):
        print(f"Error: Directory {args.images_dir} does not exist!")
        return
        
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file {args.csv_path} does not exist!")
        return
    
    analyze_dataset(args.images_dir, args.csv_path)

if __name__ == "__main__":
    main()
