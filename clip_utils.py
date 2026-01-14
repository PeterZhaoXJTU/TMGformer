import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from typing import Tuple, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from config import Config

class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        # Load local GPT2
        self.gpt = GPT2LMHeadModel.from_pretrained(Config.gpt2_model_path, local_files_only=True)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        
        if prefix_length > 10:
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))

    def forward(self, tokens, prefix, mask=None, labels=None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

class Predictor:
    def __init__(self, device):
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.prefix_length = 10
        
        # Initialize Caption Model
        self.caption_model = ClipCaptionModel(self.prefix_length)
        # Load weights
        if Config.clip_weights_path:
             self.caption_model.load_state_dict(torch.load(Config.clip_weights_path, map_location=self.device))
        
        self.caption_model = self.caption_model.eval().to(self.device)

    def predict_feature(self, image_tensor):
        """Returns the prefix embedding for loss calculation."""
        with torch.no_grad():
            # Extract features via CLIP
            prefix = self.clip_model.encode_image(image_tensor).to(self.device, dtype=torch.float32)
            # Project to GPT space
            prefix_embed = self.caption_model.clip_project(prefix)
            batch_size = prefix_embed.shape[0]
            prefix_embed = prefix_embed.view(batch_size, self.prefix_length, -1)
            return prefix_embed

def extract_clip_feature_differentiable(image_3ch_processed, clip_model):
    """Direct CLIP encoding keeping gradients if needed."""
    return F.normalize(clip_model.encode_image(image_3ch_processed), p=2, dim=-1)
