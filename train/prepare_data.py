#!/usr/bin/env python3
"""
Sample script to prepare Sentinel-1 and Sentinel-2 data for training

This script demonstrates how to:
1. Read Sentinel-1 VV and VH bands
2. Read Sentinel-2 grayscale
3. Verify co-registration
4. Create CSV file for training

Adapt this script to your data format and directory structure.
"""

import os
import csv
from pathlib import Path
import argparse


def create_training_csv(data_dir, output_csv, scene_pattern='scene_*'):
    """
    Create training CSV file from organized data directory
    
    Expected structure:
        data_dir/
        ├── scene_0001/
        │   ├── vv.tif
        │   ├── vh.tif
        │   └── s2_gray.tif
        ├── scene_0002/
        │   ├── vv.tif
        │   ├── vh.tif
        │   └── s2_gray.tif
        └── ...
    
    Args:
        data_dir: Root directory containing scenes
        output_csv: Path to output CSV file
        scene_pattern: Pattern to match scene directories
    """
    data_dir = Path(data_dir)
    scenes = sorted(data_dir.glob(scene_pattern))
    
    if not scenes:
        print(f"No scenes found matching pattern '{scene_pattern}' in {data_dir}")
        return
    
    print(f"Found {len(scenes)} scenes")
    
    # Prepare CSV data
    rows = []
    valid_scenes = 0
    
    for scene_dir in scenes:
        if not scene_dir.is_dir():
            continue
        
        # Expected file paths (adjust to your naming convention)
        vv_path = scene_dir / 'vv.tif'
        vh_path = scene_dir / 'vh.tif'
        s2_path = scene_dir / 's2_gray.tif'
        
        # Check if all files exist
        if vv_path.exists() and vh_path.exists() and s2_path.exists():
            # Store relative paths from data_dir
            rows.append({
                'vv_path': str(vv_path.relative_to(data_dir)),
                'vh_path': str(vh_path.relative_to(data_dir)),
                's2_path': str(s2_path.relative_to(data_dir))
            })
            valid_scenes += 1
        else:
            print(f"Warning: Missing files in {scene_dir.name}")
            if not vv_path.exists():
                print(f"  - Missing: vv.tif")
            if not vh_path.exists():
                print(f"  - Missing: vh.tif")
            if not s2_path.exists():
                print(f"  - Missing: s2_gray.tif")
    
    # Write CSV
    if rows:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['vv_path', 'vh_path', 's2_path']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\nCreated CSV file: {output_path}")
        print(f"Valid scenes: {valid_scenes}/{len(scenes)}")
    else:
        print("No valid scenes found. CSV not created.")


def split_train_val(csv_file, train_ratio=0.8):
    """
    Split CSV into train and validation sets
    
    Args:
        csv_file: Path to CSV file
        train_ratio: Ratio of training data (default: 0.8)
    """
    import random
    
    csv_path = Path(csv_file)
    
    # Read all rows
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Shuffle and split
    random.shuffle(rows)
    split_idx = int(len(rows) * train_ratio)
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]
    
    # Write train CSV
    train_path = csv_path.parent / 'train_pairs.csv'
    with open(train_path, 'w', newline='') as f:
        fieldnames = ['vv_path', 'vh_path', 's2_path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)
    
    # Write val CSV
    val_path = csv_path.parent / 'val_pairs.csv'
    with open(val_path, 'w', newline='') as f:
        fieldnames = ['vv_path', 'vh_path', 's2_path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(val_rows)
    
    print(f"\nSplit complete:")
    print(f"  Train: {len(train_rows)} scenes -> {train_path}")
    print(f"  Val: {len(val_rows)} scenes -> {val_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Sentinel data for SuperGlue training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data_dir', type=str,
                       help='Root directory containing scene folders')
    parser.add_argument('--output', type=str, default='data/sentinel_pairs/all_pairs.csv',
                       help='Output CSV file path')
    parser.add_argument('--pattern', type=str, default='scene_*',
                       help='Pattern to match scene directories')
    parser.add_argument('--split', action='store_true',
                       help='Split into train and validation sets')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of training data (for --split)')
    
    args = parser.parse_args()
    
    # Create CSV
    create_training_csv(args.data_dir, args.output, args.pattern)
    
    # Split if requested
    if args.split and Path(args.output).exists():
        split_train_val(args.output, args.train_ratio)


if __name__ == '__main__':
    main()
