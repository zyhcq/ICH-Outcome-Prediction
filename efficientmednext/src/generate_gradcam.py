#!/usr/bin/env python3
"""
SEG-GRAD-CAM Generator
"""
import os
import sys
import argparse
import json
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

import monai.transforms as mt
from monai.data import MetaTensor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
REPO_DIR = os.path.join(PROJECT_DIR, 'EfficientMedNeXt')
sys.path.insert(0, REPO_DIR)

from networks.MedNeXt.mednextv1.create_efficient_mednext import create_efficient_mednext

def setup_logger(name, log_file=None):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description='SEG-GRAD-CAM Explainability Heatmap Generator')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--weights_path', type=str, default='checkpoints/stage2_finetune/finetune_best.pth')
    parser.add_argument('--output_dir', type=str, default='gradcam_results')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='test')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[160, 160, 32])
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--save_3d_npy', action='store_true')
    return parser.parse_args()

def build_data_list(data_dir, split_type, fold=0):
    data_list = []
    if split_type == 'val':
        splits_path = os.path.join(data_dir, 'splits_final.json')
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        if isinstance(splits, list):
            splits = splits[fold]
        for pid in splits['val']:
            img = os.path.join(data_dir, 'images', 'imagesTr', f'{pid}_0000.nii.gz')
            msk = os.path.join(data_dir, 'masks', 'labelsTr', f'{pid}.nii.gz')
            if os.path.exists(img) and os.path.exists(msk):
                data_list.append({'image': img, 'mask': msk, 'pid': pid})
    elif split_type == 'test':
        img_dir = os.path.join(data_dir, 'images', 'imagesTs')
        mask_dir = os.path.join(data_dir, 'masks', 'labelsTs')
        for f in sorted(os.listdir(img_dir)):
            if f.endswith('_0000.nii.gz'):
                pid = f.replace('_0000.nii.gz', '')
                img = os.path.join(img_dir, f)
                msk = os.path.join(mask_dir, f"{pid}.nii.gz")
                if os.path.exists(msk):
                    data_list.append({'image': img, 'mask': msk, 'pid': pid})
    return data_list

def get_inference_transforms():
    return mt.Compose([
        mt.LoadImaged(keys=['image', 'mask'], image_only=True),
        mt.EnsureChannelFirstd(keys=['image', 'mask']),
        mt.Spacingd(keys=['image', 'mask'], pixdim=(0.5, 0.5, 5.0), mode=('bilinear', 'nearest')),
        mt.ScaleIntensityRanged(keys=['image'], a_min=0, a_max=80, b_min=0, b_max=80, clip=True),
        mt.NormalizeIntensityd(keys=['image'], nonzero=False),
    ])


class SegGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_patch(self, patch_tensor, class_idx=1):
        self.model.zero_grad()
        with autocast():
            output = self.model(patch_tensor)
        target = output[0, class_idx].float().sum()
        target.backward()

        weights = self.gradients[0].float().mean(dim=[1, 2, 3])
        cam = (weights[:, None, None, None] * self.activations[0].float()).sum(dim=0)
        cam = F.relu(cam)

        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=patch_tensor.shape[2:], mode='trilinear', align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        return cam


def generate_cam_sliding_window(model, seg_cam, img_tensor, patch_size, device):
    _, _, H, W, D = img_tensor.shape
    pH, pW, pD = patch_size
    
    cam_full = np.zeros((H, W, D), dtype=np.float32)
    count_map = np.zeros((H, W, D), dtype=np.float32)

    for h_start in range(0, H, pH):
        for w_start in range(0, W, pW):
            for d_start in range(0, D, pD):
                h_end = min(h_start + pH, H)
                w_end = min(w_start + pW, W)
                d_end = min(d_start + pD, D)

                patch = img_tensor[:, :, h_start:h_end, w_start:w_end, d_start:d_end]
                pad_h = pH - (h_end - h_start)
                pad_w = pW - (w_end - w_start)
                pad_d = pD - (d_end - d_start)

                if pad_h > 0 or pad_w > 0 or pad_d > 0:
                    patch = F.pad(patch, (0, pad_d, 0, pad_w, 0, pad_h), mode='constant', value=0)

                patch = patch.to(device)
                cam_patch = seg_cam.generate_patch(patch, class_idx=1)

                actual_h = h_end - h_start
                actual_w = w_end - w_start
                actual_d = d_end - d_start
                cam_patch = cam_patch[:actual_h, :actual_w, :actual_d]

                cam_full[h_start:h_end, w_start:w_end, d_start:d_end] += cam_patch
                count_map[h_start:h_end, w_start:w_end, d_start:d_end] += 1.0

                del patch
                torch.cuda.empty_cache()

    count_map[count_map == 0] = 1.0
    cam_full /= count_map

    if cam_full.max() > 0:
        cam_full /= cam_full.max()

    return cam_full


def find_best_slice(mask_3d):
    if mask_3d.ndim == 4:
        mask_3d = mask_3d[0]
    slice_areas = mask_3d.sum(axis=(0, 1))
    return int(np.argmax(slice_areas))

def apply_ct_window(ct_slice, window_center=40, window_width=80):
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    ct_slice = np.clip(ct_slice, lower, upper)
    return (ct_slice - lower) / (upper - lower)

def draw_quad_figure(ct_vol, gt_mask, pred_mask, cam_map, pid, output_path, dpi=150):
    z_idx = find_best_slice(gt_mask)
    ct_slice = ct_vol[0, :, :, z_idx] if ct_vol.ndim == 4 else ct_vol[:, :, z_idx]
    gt_slice = gt_mask[0, :, :, z_idx] if gt_mask.ndim == 4 else gt_mask[:, :, z_idx]
    pred_slice = pred_mask[:, :, z_idx]
    cam_slice = cam_map[:, :, z_idx]

    ct_display = apply_ct_window(ct_slice, window_center=40, window_width=80)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=dpi)
    fig.suptitle(f'{pid}  |  Slice Z={z_idx}', fontsize=14, fontweight='bold')

    axes[0, 0].imshow(ct_display.T, cmap='gray', origin='lower')
    axes[0, 0].set_title('CT (Brain Window)', fontsize=11)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(ct_display.T, cmap='gray', origin='lower')
    gt_overlay = np.ma.masked_where(gt_slice.T == 0, gt_slice.T)
    axes[0, 1].imshow(gt_overlay, cmap='autumn', alpha=0.5, origin='lower', vmin=0, vmax=1)
    axes[0, 1].set_title('Ground Truth', fontsize=11)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(ct_display.T, cmap='gray', origin='lower')
    pred_overlay = np.ma.masked_where(pred_slice.T == 0, pred_slice.T)
    axes[1, 0].imshow(pred_overlay, cmap='Greens', alpha=0.5, origin='lower', vmin=0, vmax=1)
    axes[1, 0].set_title('Prediction', fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(ct_display.T, cmap='gray', origin='lower')
    axes[1, 1].imshow(cam_slice.T, cmap='jet', alpha=0.4, origin='lower', vmin=0, vmax=1)
    axes[1, 1].set_title('SEG-GRAD-CAM', fontsize=11)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('gradcam', os.path.join(args.output_dir, 'gradcam.log'))
    logger.info(f"Args: {args}")
    device = torch.device(args.device)

    data_list = build_data_list(args.data_dir, args.split, args.fold)
    if not data_list:
        logger.error("No data found.")
        return
    logger.info(f"Loaded {len(data_list)} subjects for {args.split} Grad-CAM.")

    transforms = get_inference_transforms()

    model = create_efficient_mednext(
        num_input_channels=1, num_classes=2, model_id='L',
        n_channels=32, kernel_sizes=[1, 3, 5], strides=[1, 1, 1],
        uniform_dec_channels=32, deep_supervision=False, mode='val'
    ).to(device)

    ckpt = torch.load(args.weights_path, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in ckpt.get('network_weights', ckpt).items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    seg_cam = SegGradCAM(model, model.dec_block_0)
    logger.info("SEG-GRAD-CAM attached to dec_block_0")

    fig_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    if args.save_3d_npy:
        npy_dir = os.path.join(args.output_dir, 'heatmaps_3d')
        os.makedirs(npy_dir, exist_ok=True)

    patch_size = tuple(args.patch_size)

    for item in tqdm(data_list, desc="Generating Grad-CAM"):
        pid = item['pid']
        data = transforms({'image': item['image'], 'mask': item['mask']})
        img_tensor = data['image'].unsqueeze(0)
        gt_mask = data['mask'].cpu().numpy()

        cam = generate_cam_sliding_window(model, seg_cam, img_tensor, patch_size, device)

        with torch.no_grad():
            from monai.inferers import sliding_window_inference
            pred_logits = sliding_window_inference(
                inputs=img_tensor.to(device),
                roi_size=patch_size,
                sw_batch_size=4,
                predictor=model,
                overlap=0.25,
                mode="constant"
            )
            pred_mask = torch.argmax(pred_logits, dim=1, keepdim=True)[0, 0].cpu().numpy().astype(float)
            del pred_logits
            torch.cuda.empty_cache()

        ct_raw = data['image'].cpu().numpy()
        ct_for_display = ct_raw * 20.0 + 20.0

        draw_quad_figure(ct_for_display, gt_mask, pred_mask, cam, pid,
                         os.path.join(fig_dir, f'{pid}.png'), args.dpi)

        if args.save_3d_npy:
            np.save(os.path.join(npy_dir, f'{pid}_dec0.npy'), cam.astype(np.float16))

        del img_tensor, cam, pred_mask, gt_mask
        torch.cuda.empty_cache()

    logger.info(f"Done! {len(data_list)} Grad-CAM figures saved to {fig_dir}")

if __name__ == '__main__':
    main()
