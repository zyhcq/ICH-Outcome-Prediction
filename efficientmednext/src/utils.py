"""
Utility Functions
"""
import os
import random
import logging
import numpy as np
import torch
import torch.distributed as dist

def setup_logger(name, log_file=None, level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

class InfiniteRandomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, seed=0):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
            yield from indices
            self.epoch += 1

    def __len__(self):
        return len(self.dataset)

class InfiniteDistributedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available(): raise RuntimeError("Requires distributed")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available(): raise RuntimeError("Requires distributed")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        import math
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            if self.shuffle:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))
            indices += indices[:(self.total_size - len(indices))]
            indices = indices[self.rank:self.total_size:self.num_replicas]
            yield from indices
            self.epoch += 1

    def __len__(self):
        return self.num_samples

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_main_process():
    if not dist.is_initialized(): return True
    return dist.get_rank() == 0

def plot_progress(history, output_dir, filename='progress.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(font_scale=2.5)
    except ImportError:
        plt.rcParams.update({'font.size': 18})

    epoch = len(history['train_loss']) - 1
    x_values = list(range(epoch + 1))
    fig, ax_all = plt.subplots(3, 1, figsize=(30, 54))

    ax = ax_all[0]
    ax2 = ax.twinx()
    ax.plot(x_values, history['train_loss'][:epoch + 1], color='b', ls='-', label='loss_tr', linewidth=4)
    ax.plot(x_values, history['val_loss'][:epoch + 1], color='r', ls='-', label='loss_val', linewidth=4)
    if 'mean_fg_dice' in history:
        ax2.plot(x_values, history['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label='pseudo dice', linewidth=3)
    if 'ema_fg_dice' in history:
        ax2.plot(x_values, history['ema_fg_dice'][:epoch + 1], color='g', ls='-', label='pseudo dice (mov. avg.)', linewidth=4)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax2.set_ylabel('pseudo dice')
    ax.legend(loc=(0, 1))
    ax2.legend(loc=(0.2, 1))

    ax = ax_all[1]
    if 'epoch_duration' in history:
        ax.plot(x_values, history['epoch_duration'][:epoch + 1], color='b', ls='-', label='epoch duration', linewidth=4)
        ylim = [0, ax.get_ylim()[1]]
        ax.set(ylim=ylim)
    ax.set_xlabel('epoch')
    ax.set_ylabel('time [s]')
    ax.legend(loc=(0, 1))

    ax = ax_all[2]
    ax.plot(x_values, history['lr'][:epoch + 1], color='b', ls='-', label='learning rate', linewidth=4)
    ax.set_xlabel('epoch')
    ax.set_ylabel('learning rate')
    ax.legend(loc=(0, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=100)
    plt.close(fig)
