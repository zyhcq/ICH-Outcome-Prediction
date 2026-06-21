#!/usr/bin/env python3
"""
Stage 2: ICH Segmentation Fine-tuning
Execution strategy: load pure weights, large patch size (160x160x32), AdamW, Linear Warmup.
"""
import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
REPO_DIR = os.path.join(PROJECT_DIR, 'EfficientMedNeXt')
sys.path.insert(0, REPO_DIR)

from src.dataset import create_datasets
from src.losses import DeepSupervisionLoss
from src.utils import set_seed, setup_logger, is_main_process, plot_progress, InfiniteRandomSampler, InfiniteDistributedSampler

def parse_args():
    parser = argparse.ArgumentParser(description='Stage 2: ICH Segmentation Fine-tuning')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--pretrained_weights', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage2_finetune')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--warmup_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patch_size', type=int, nargs=3, default=[160, 160, 32])
    parser.add_argument('--num_iterations_per_epoch', type=int, default=250)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--preprocessed_dir', type=str, default=None)
    return parser.parse_args()

def get_tp_fp_fn(pred_logits, target):
    pred = torch.argmax(pred_logits, dim=1, keepdim=True).float()
    target = target.float()
    axes = list(range(2, pred.ndim))
    tp = (pred * target).sum(dim=axes).sum(dim=0)
    fp = (pred * (1 - target)).sum(dim=axes).sum(dim=0)
    fn = ((1 - pred) * target).sum(dim=axes).sum(dim=0)
    return tp, fp, fn

def save_checkpoint(model, optimizer, scaler, best_ema, history, epoch, path):
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'epoch': epoch,
        'network_weights': model_state,
        'optimizer_state': optimizer.state_dict(),
        'grad_scaler_state': scaler.state_dict(),
        'best_ema': best_ema,
        'logging': history,
    }, path)

def get_lr_lambda(current_epoch, warmup_epochs, total_epochs):
    if current_epoch < warmup_epochs:
        return float(current_epoch + 1) / float(max(1, warmup_epochs))
    else:
        return (1.0 - (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)) ** 0.9

def main():
    args = parse_args()
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        dist.init_process_group(backend='nccl' if sys.platform != 'win32' else 'gloo')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    set_seed(args.seed + local_rank)

    logger = None
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        logger = setup_logger('finetune_seg', os.path.join(args.output_dir, 'finetune.log'))
        logger.info(f'Finetuning Args: {args}')

    train_ds, val_ds = create_datasets(
        data_dir=args.data_dir, fold=args.fold,
        patch_size=tuple(args.patch_size),
        preprocessed_dir=args.preprocessed_dir,
    )

    if world_size > 1:
        train_sampler = InfiniteDistributedSampler(train_ds, seed=args.seed)
        val_sampler = InfiniteDistributedSampler(val_ds, seed=args.seed + 1000)
    else:
        train_sampler = InfiniteRandomSampler(train_ds, seed=args.seed)
        val_sampler = InfiniteRandomSampler(val_ds, seed=args.seed + 1000)

    import math
    samples_per_gpu = math.ceil(len(val_ds) / max(world_size, 1))
    num_val_iters = math.ceil(samples_per_gpu / args.batch_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    from networks.MedNeXt.mednextv1.create_efficient_mednext import create_efficient_mednext
    model = create_efficient_mednext(
        num_input_channels=1, num_classes=2, model_id='L',
        n_channels=32, kernel_sizes=[1, 3, 5], strides=[1, 1, 1],
        uniform_dec_channels=32, deep_supervision=True, mode='train',
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        model._set_static_graph()

    if not os.path.exists(args.pretrained_weights):
        raise FileNotFoundError(f"Pretrained weights not found at {args.pretrained_weights}")
    
    if is_main_process():
        logger.info(f"Loading strictly network weights from {args.pretrained_weights}")
    
    checkpoint = torch.load(args.pretrained_weights, map_location=device)
    state_dict = checkpoint.get('network_weights', checkpoint)
    
    if world_size > 1:
        model.module.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)

    criterion = DeepSupervisionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: get_lr_lambda(ep, args.warmup_epochs, args.epochs))
    scaler = GradScaler()

    history = {'train_loss': [], 'val_loss': [], 'mean_fg_dice': [], 'ema_fg_dice': [], 'lr': [], 'epoch_duration': []}
    best_ema = None
    train_iter, val_iter = iter(train_loader), iter(val_loader)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        train_losses = []

        for _ in range(args.num_iterations_per_epoch):
            batch = next(train_iter)
            data, target = batch['image'].to(device, non_blocking=True), batch['mask'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                output = model(data, mode='train')
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.detach().cpu().numpy())

        model.eval()
        val_losses = []
        tp_sum, fp_sum, fn_sum = np.zeros(1), np.zeros(1), np.zeros(1)

        with torch.no_grad():
            for _ in range(num_val_iters):
                batch = next(val_iter)
                data, target = batch['image'].to(device, non_blocking=True), batch['mask'].to(device, non_blocking=True)
                with autocast():
                    output = model(data, mode='train')
                    loss = criterion(output, target)
                    pred = output[0] if isinstance(output, (list, tuple)) else output
                tp, fp, fn = get_tp_fp_fn(pred, target)
                val_losses.append(loss.detach().cpu().numpy())
                tp_sum += tp.cpu().numpy()
                fp_sum += fp.cpu().numpy()
                fn_sum += fn.cpu().numpy()

        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size()
            tl_g, vl_g = [None]*ws, [None]*ws
            tp_g, fp_g, fn_g = [None]*ws, [None]*ws, [None]*ws
            dist.all_gather_object(tl_g, train_losses)
            dist.all_gather_object(vl_g, val_losses)
            dist.all_gather_object(tp_g, tp_sum)
            dist.all_gather_object(fp_g, fp_sum)
            dist.all_gather_object(fn_g, fn_sum)
            
            t_loss = np.mean([l for s in tl_g for l in s])
            v_loss = np.mean([l for s in vl_g for l in s])
            tp_total = np.sum(tp_g, axis=0)
            fp_total = np.sum(fp_g, axis=0)
            fn_total = np.sum(fn_g, axis=0)
        else:
            t_loss = np.mean(train_losses)
            v_loss = np.mean(val_losses)
            tp_total, fp_total, fn_total = tp_sum, fp_sum, fn_sum

        pseudo_dice = float(2 * tp_total / (2 * tp_total + fp_total + fn_total + 1e-8))
        ema_dice = pseudo_dice if not history['ema_fg_dice'] else 0.9 * history['ema_fg_dice'][-1] + 0.1 * pseudo_dice

        if is_main_process():
            history['train_loss'].append(float(t_loss))
            history['val_loss'].append(float(v_loss))
            history['mean_fg_dice'].append(pseudo_dice)
            history['ema_fg_dice'].append(ema_dice)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['epoch_duration'].append(time.time() - epoch_start)

            logger.info(f"Finetune Epoch {epoch} loss: {t_loss:.4f} | val loss: {v_loss:.4f} | Dice: {pseudo_dice:.4f} | EMA: {ema_dice:.4f}")

            if (epoch + 1) % args.save_every == 0 and epoch != args.epochs - 1:
                save_checkpoint(model, optimizer, scaler, best_ema, history, epoch + 1, os.path.join(args.output_dir, 'finetune_latest.pth'))

            if best_ema is None or ema_dice > best_ema:
                best_ema = ema_dice
                logger.info(f"New best EMA: {best_ema:.4f}")
                save_checkpoint(model, optimizer, scaler, best_ema, history, epoch + 1, os.path.join(args.output_dir, 'finetune_best.pth'))

            plot_progress(history, args.output_dir)

        scheduler.step()

    if is_main_process():
        save_checkpoint(model, optimizer, scaler, best_ema, history, args.epochs, os.path.join(args.output_dir, 'finetune_final.pth'))
        latest_path = os.path.join(args.output_dir, 'finetune_latest.pth')
        if os.path.exists(latest_path): os.remove(latest_path)
    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == '__main__':
    main()
