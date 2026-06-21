#!/usr/bin/env python3
"""
Offline Preprocessing Script
"""
import os
import time
import argparse
import numpy as np
import monai.transforms as mt

def parse_args():
    parser = argparse.ArgumentParser(description='Offline preprocessing')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='preprocessed')
    return parser.parse_args()

def get_preprocess_transforms():
    return mt.Compose([
        mt.LoadImaged(keys=['image', 'mask'], image_only=True),
        mt.EnsureChannelFirstd(keys=['image', 'mask']),
        mt.Spacingd(keys=['image', 'mask'], pixdim=(0.5, 0.5, 5.0), mode=('bilinear', 'nearest')),
        mt.ScaleIntensityRanged(keys=['image'], a_min=0, a_max=80, b_min=0, b_max=80, clip=True),
        mt.CropForegroundd(keys=['image', 'mask'], source_key='image', margin=10),
        mt.NormalizeIntensityd(keys=['image'], nonzero=False),
    ])

def main():
    args = parse_args()

    images_dir = os.path.join(args.data_dir, 'images', 'imagesTr')
    masks_dir = os.path.join(args.data_dir, 'masks', 'labelsTr')
    
    all_ids = [f.replace('_0000.nii.gz', '') for f in os.listdir(images_dir) if f.endswith('_0000.nii.gz')]
    
    print(f'Found {len(all_ids)} samples to preprocess')

    out_img_dir = os.path.join(args.output_dir, 'imagesTr')
    out_msk_dir = os.path.join(args.output_dir, 'labelsTr')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)

    transforms = get_preprocess_transforms()
    t_start = time.time()
    
    for i, pid in enumerate(all_ids):
        img_path = os.path.join(images_dir, f'{pid}_0000.nii.gz')
        msk_path = os.path.join(masks_dir, f'{pid}.nii.gz')
        if not os.path.exists(msk_path):
            print(f'  SKIP {pid}: mask file not found')
            continue

        data = transforms({'image': img_path, 'mask': msk_path})

        img_np = data['image'].numpy().astype(np.float32)
        msk_raw = data['mask'].numpy()
        msk_np = (msk_raw > 0.5).astype(np.uint8)

        np.save(os.path.join(out_img_dir, f'{pid}.npy'), img_np)
        np.save(os.path.join(out_msk_dir, f'{pid}.npy'), msk_np)

        if (i + 1) % 50 == 0 or (i + 1) == len(all_ids):
            elapsed = time.time() - t_start
            speed = (i + 1) / elapsed
            print(f'  [{i+1}/{len(all_ids)}] {pid} shape={img_np.shape} ({speed:.1f} it/s)')

    print(f'\nDone preprocessing in {time.time() - t_start:.1f}s')

if __name__ == '__main__':
    main()
