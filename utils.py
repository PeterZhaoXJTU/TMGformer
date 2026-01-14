import torch
import torch.nn as nn
from torchvision.transforms import Resize, Normalize, InterpolationMode
import numpy as np

# --- Differentiable Visualizer (Dynamic Quantile) ---
class DifferentiableDynamicVisualizer(nn.Module):
    def __init__(self,
                 num_input_channels: int,
                 channels_to_select: list = [0, 1, 2], # R, G, B
                 output_order: list = [2, 1, 0],      # BGR output
                 tol_percent: tuple = (0.03, 0.97)    # Percentiles for stretching
                 ):
        super().__init__()
        self.channels_to_select = channels_to_select
        self.output_order = output_order
        self.tol_percent = tol_percent
        print(f"Initialized DifferentiableDynamicVisualizer: Channels {channels_to_select}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies channel selection, DYNAMIC percentile stretch, and reordering.
        Differentiable w.r.t input.
        """
        B, _, H, W = image.shape
        selected_img = image[:, self.channels_to_select, :, :] 
        stretched_img = torch.zeros_like(selected_img)

        for b in range(B):
            for c in range(3):
                channel_data = selected_img[b, c, :, :]
                channel_flat = channel_data.view(-1)

                try:
                    t1 = torch.quantile(channel_flat.float(), self.tol_percent[0])
                    t2 = torch.quantile(channel_flat.float(), self.tol_percent[1])
                except RuntimeError:
                    t1, t2 = torch.min(channel_flat), torch.max(channel_flat)

                denominator = t2 - t1 + 1e-8
                channel_clamped = torch.clamp(channel_data, min=t1, max=t2)
                stretched_channel = (channel_clamped - t1) / denominator
                stretched_img[b, c, :, :] = torch.clamp(stretched_channel, 0.0, 1.0)

        return stretched_img[:, self.output_order, :, :]

# --- CLIP Preprocessing ---
class DifferentiableCLIPPreprocess(nn.Module):
    def __init__(self, target_size=224, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
        super().__init__()
        self.resize = Resize((target_size, target_size), interpolation=InterpolationMode.BICUBIC, antialias=True)
        self.normalize = Normalize(mean=torch.tensor(mean), std=torch.tensor(std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resize(x)
        x = self.normalize(x)
        return x

# --- 4. Logging Helper ---
def write_list_to_txt(lst, filename):
    """Writes a list of items to a text file, appending to it."""
    with open(filename, 'a') as file:
        for item in lst:
            file.write(str(item) + '\n')
