"""
Dataset Loaders (v2 refactored)
DRY implementation, natively reads nnUNet splits_final.json
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import monai.transforms as mt
from monai.data import MetaTensor

def get_base_augmentations():
    return [
        mt.RandAffined(
            keys=['image', 'mask'], prob=0.2, rotate_range=[0.5236] * 3,
            scale_range=[(-0.3, 0.4)] * 3, mode=('bilinear', 'nearest'),
            padding_mode='border',
        ),
        mt.RandGaussianNoised(keys=['image'], prob=0.1, mean=0.0, std=0.1),
        mt.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
        mt.RandScaleIntensityd(keys=['image'], factors=0.25, prob=0.15),
        mt.RandAdjustContrastd(keys=['image'], prob=0.15, gamma=(0.75, 1.25)),
        mt.RandSimulateLowResolutiond(keys=['image'], prob=0.25, zoom_range=(0.5, 1.0)),
        mt.RandAdjustContrastd(keys=['image'], prob=0.1, gamma=(0.7, 1.5), invert_image=True, retain_stats=True),
        mt.RandAdjustContrastd(keys=['image'], prob=0.3, gamma=(0.7, 1.5), retain_stats=True),
        mt.RandFlipd(keys=['image', 'mask'], spatial_axis=[0], prob=0.5),
        mt.RandFlipd(keys=['image', 'mask'], spatial_axis=[1], prob=0.5),
        mt.RandFlipd(keys=['image', 'mask'], spatial_axis=[2], prob=0.5),
    ]

def get_train_transforms(patch_size):
    return mt.Compose([
        mt.LoadImaged(keys=['image', 'mask'], image_only=True),
        mt.EnsureChannelFirstd(keys=['image', 'mask']),
        mt.Spacingd(keys=['image', 'mask'], pixdim=(0.5, 0.5, 5.0), mode=('bilinear', 'nearest')),
        mt.ScaleIntensityRanged(keys=['image'], a_min=0, a_max=80, b_min=0, b_max=80, clip=True),
        mt.CropForegroundd(keys=['image', 'mask'], source_key='image', margin=10),
        mt.NormalizeIntensityd(keys=['image'], nonzero=False),
        mt.SpatialPadd(keys=['image', 'mask'], spatial_size=patch_size),
        mt.RandCropByPosNegLabeld(
            keys=['image', 'mask'], label_key='mask', spatial_size=patch_size,
            pos=1, neg=2, num_samples=1, image_key='image', allow_smaller=False,
        ),
    ] + get_base_augmentations())

def get_val_transforms(patch_size):
    return mt.Compose([
        mt.LoadImaged(keys=['image', 'mask'], image_only=True),
        mt.EnsureChannelFirstd(keys=['image', 'mask']),
        mt.Spacingd(keys=['image', 'mask'], pixdim=(0.5, 0.5, 5.0), mode=('bilinear', 'nearest')),
        mt.ScaleIntensityRanged(keys=['image'], a_min=0, a_max=80, b_min=0, b_max=80, clip=True),
        mt.CropForegroundd(keys=['image', 'mask'], source_key='image', margin=10),
        mt.NormalizeIntensityd(keys=['image'], nonzero=False),
        mt.SpatialPadd(keys=['image', 'mask'], spatial_size=patch_size),
        mt.RandCropByPosNegLabeld(
            keys=['image', 'mask'], label_key='mask', spatial_size=patch_size,
            pos=1, neg=1, num_samples=1, image_key='image', allow_smaller=False,
        ),
    ])

def get_train_transforms_preprocessed(patch_size):
    return mt.Compose([
        mt.SpatialPadd(keys=['image', 'mask'], spatial_size=patch_size),
        mt.RandCropByPosNegLabeld(
            keys=['image', 'mask'], label_key='mask', spatial_size=patch_size,
            pos=1, neg=2, num_samples=1, image_key='image', allow_smaller=False,
        ),
    ] + get_base_augmentations())

def get_val_transforms_preprocessed(patch_size):
    return mt.Compose([
        mt.SpatialPadd(keys=['image', 'mask'], spatial_size=patch_size),
        mt.RandCropByPosNegLabeld(
            keys=['image', 'mask'], label_key='mask', spatial_size=patch_size,
            pos=1, neg=1, num_samples=1, image_key='image', allow_smaller=False,
        ),
    ])

class PreprocessedICHDataset(Dataset):
    def __init__(self, data_list, transforms):
        self.data_list = data_list
        self.transforms = transforms
        self.affine = torch.diag(torch.tensor([0.5, 0.5, 5.0, 1.0], dtype=torch.float32))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img = np.load(item['image'])
        msk = np.load(item['mask'])
        data = {
            'image': MetaTensor(torch.from_numpy(img).float(), affine=self.affine),
            'mask': MetaTensor(torch.from_numpy(msk.astype(np.float32)), affine=self.affine),
        }
        res = self.transforms(data)
        if isinstance(res, list): res = res[0]
        return {'image': res['image'], 'mask': res['mask'], 'pid': item['pid']}

class ICHDataset(Dataset):
    def __init__(self, data_list, transforms):
        self.data_list = data_list
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        res = self.transforms({'image': item['image'], 'mask': item['mask']})
        if isinstance(res, list): res = res[0]
        return {'image': res['image'], 'mask': res['mask'], 'pid': item['pid']}

def build_data_list(base_dir, split_ids, is_preprocessed=False):
    data_list = []
    for pid in split_ids:
        if is_preprocessed:
            img_path = os.path.join(base_dir, 'imagesTr', f'{pid}.npy')
            mask_path = os.path.join(base_dir, 'labelsTr', f'{pid}.npy')
        else:
            img_path = os.path.join(base_dir, 'images', 'imagesTr', f'{pid}_0000.nii.gz')
            mask_path = os.path.join(base_dir, 'masks', 'labelsTr', f'{pid}.nii.gz')
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            data_list.append({'image': img_path, 'mask': mask_path, 'pid': pid})
    return data_list

def create_datasets(data_dir, fold=0, patch_size=(192, 192, 32), preprocessed_dir=None):
    splits_path = os.path.join(data_dir, 'splits_final.json')
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Missing nnUNet splits_final.json in {splits_path}")

    with open(splits_path, 'r') as f:
        splits = json.load(f)
    if isinstance(splits, list):
        splits = splits[fold]

    if preprocessed_dir:
        train_list = build_data_list(preprocessed_dir, splits['train'], is_preprocessed=True)
        val_list = build_data_list(preprocessed_dir, splits['val'], is_preprocessed=True)
        return PreprocessedICHDataset(train_list, get_train_transforms_preprocessed(patch_size)), \
               PreprocessedICHDataset(val_list, get_val_transforms_preprocessed(patch_size))
    else:
        train_list = build_data_list(data_dir, splits['train'], is_preprocessed=False)
        val_list = build_data_list(data_dir, splits['val'], is_preprocessed=False)
        return ICHDataset(train_list, get_train_transforms(patch_size)), \
               ICHDataset(val_list, get_val_transforms(patch_size))
