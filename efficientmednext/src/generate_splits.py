#!/usr/bin/env python3
"""
nnU-Net style 5-fold split generation script
Reads all samples from raw_data/imagesTr and generates standard splits_final.json.
Ensures patients in training and validation sets are strictly disjoint.
"""
import os
import json
import argparse
from sklearn.model_selection import KFold

def parse_args():
    parser = argparse.ArgumentParser(description='Generate 5-fold splits for nnU-Net pipeline')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to raw_data directory containing imagesTr')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds to split the dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    args = parse_args()
    
    images_dir = os.path.join(args.data_dir, 'imagesTr')
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Directory not found: {images_dir}")
        
    file_list = [f for f in os.listdir(images_dir) if f.endswith('_0000.nii.gz')]
    patient_ids = sorted([f.replace('_0000.nii.gz', '') for f in file_list])
    
    if not patient_ids:
        raise ValueError(f"No valid image files found in {images_dir}. Ensure files end with '_0000.nii.gz'")
        
    print(f"Found {len(patient_ids)} cases. Splitting into {args.num_folds} folds...")
    
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    
    splits = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        train_pids = [patient_ids[i] for i in train_idx]
        val_pids = [patient_ids[i] for i in val_idx]
        
        splits.append({
            "train": list(train_pids),
            "val": list(val_pids)
        })
        print(f"Fold {fold}: {len(train_pids)} train, {len(val_pids)} val")

    output_path = os.path.join(args.data_dir, 'splits_final.json')
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=4)
        
    print(f"Successfully saved {args.num_folds}-fold split to {output_path}")

if __name__ == '__main__':
    main()
