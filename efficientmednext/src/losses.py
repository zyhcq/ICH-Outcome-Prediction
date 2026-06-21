"""
Loss Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss

class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision weighted loss wrapper
    Uses MONAI's DiceCELoss (Softmax) as base loss
    """
    def __init__(self, weights=None, smooth=1e-5):
        super().__init__()
        self.base_loss = DiceCELoss(softmax=True, to_onehot_y=True, squared_pred=True, smooth_nr=smooth, smooth_dr=smooth)
        weights = weights or [1.0, 0.5, 0.25, 0.125, 0.0625]
        weights[-1] = 1e-6
        weight_sum = sum(weights)
        self.weights = [w / weight_sum for w in weights]

    def forward(self, outputs, target):
        if not isinstance(outputs, (list, tuple)):
            return self.base_loss(outputs, target.float())

        total_loss = 0.0
        for i, pred in enumerate(outputs):
            if i >= len(self.weights):
                break
            if pred.shape[2:] != target.shape[2:]:
                t = F.interpolate(target.float(), size=pred.shape[2:], mode='nearest')
            else:
                t = target.float()
            total_loss += self.weights[i] * self.base_loss(pred, t)
        return total_loss
