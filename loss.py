import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np

# --- Helper Functions ---
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_func(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel)
    if img1.is_cuda: window = window.to(img1.device)
    window = window.type_as(img1)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

# --- Loss Modules ---

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
    def forward(self, img1, img2):
        return 1 - ssim_func(img1, img2, self.window_size, self.size_average)

def cosine_similarity_loss_normalized(feature1, feature2, pool_type="mean"):
    if pool_type == "mean":
        f1, f2 = feature1.mean(dim=1), feature2.mean(dim=1)
    elif pool_type == "max":
        f1, f2 = feature1.max(dim=1)[0], feature2.max(dim=1)[0]
    else: raise ValueError("Pool type must be mean or max")
    
    f1 = F.normalize(f1, p=2, dim=-1)
    f2 = F.normalize(f2, p=2, dim=-1)
    return 1.0 - (f1 * f2).sum(dim=-1).mean()

class Gauss_filter(nn.Module):
    def __init__(self, channels):
        super(Gauss_filter, self).__init__()
        kernel = torch.tensor([[0.0265, 0.0354, 0.0390, 0.0354, 0.0265],
                               [0.0354, 0.0473, 0.0520, 0.0473, 0.0354],
                               [0.0390, 0.0520, 0.0573, 0.0520, 0.0390],
                               [0.0354, 0.0473, 0.0520, 0.0473, 0.0354],
                               [0.0265, 0.0354, 0.0390, 0.0354, 0.0265]])
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
    def forward(self, x):
        return F.conv2d(x, self.weight, padding=2, groups=x.shape[1])

class InfoNCELossWithGT(nn.Module):
    def __init__(self, temperature=0.07, channels=4):
        super(InfoNCELossWithGT, self).__init__()
        self.temperature = temperature
        self.low = Gauss_filter(channels)

    def forward(self, left_features, right_features, gt_features):
        n = gt_features.shape[0]
        gt_features_l = self.low(gt_features).view(n, -1)
        left_features = self.low(left_features).view(n, -1)
        right_features = right_features.view(n, -1)
        
        left_norm = F.normalize(left_features, p=2, dim=1)
        right_norm = F.normalize(right_features, p=2, dim=1)
        gt_norm_l = F.normalize(gt_features_l, p=2, dim=1)
        
        pos_logits = (right_norm * gt_norm_l).sum(dim=1, keepdim=True) / self.temperature
        neg_logits = torch.matmul(right_norm, left_norm.t()) / self.temperature
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(len(right_features), dtype=torch.long, device=right_features.device)
        
        return F.cross_entropy(logits, labels)
