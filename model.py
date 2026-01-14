import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

# --- Basic Blocks ---

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral): normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral): normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type =='BiasFree' else WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return rearrange(self.body(rearrange(x, 'b c h w -> b (h w) c')), 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm_noss(nn.Module):
     def __init__(self, dim):
         super().__init__()
         self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
     def forward(self, x):
         h, w = x.shape[-2:]
         x = rearrange(x, 'b c h w -> b (h w) c')
         x = self.norm(x)
         return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
    def forward(self, x):
        return self.proj(x)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = (attn.softmax(dim=-1) @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, padding=1, bias=False), nn.PixelUnshuffle(2))
    def forward(self, x): return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, padding=1, bias=False), nn.PixelShuffle(2))
    def forward(self, x): return self.body(x)

# --- Advanced Modules for DiT ---

def modulate2d_0627(x, shift, scale):
    return x * (1 + scale) + shift

class TransformerDiT_0627(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerDiT_0627, self).__init__()
        self.norm1 = LayerNorm_noss(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm_noss(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Conv2d(dim, 6 * dim, kernel_size=1, bias=True))

    def forward(self, x_in):
        x, text = x_in[0], x_in[1]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(text).chunk(6, dim=1)
        x = x + gate_msa * self.attn(modulate2d_0627(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.ffn(modulate2d_0627(self.norm2(x), shift_mlp, scale_mlp))
        return [x, text]

class TransformerDiT(nn.Module):
    def __init__(self, dim, hidden_size ,num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerDiT, self).__init__()
        self.norm1 = LayerNorm_noss(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm_noss(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * dim, bias=True))
        
    def modulate2d(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x_in):
        x, text = x_in[0], x_in[1]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(text.mean(dim=1)).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(-1).unsqueeze(-1) * self.attn(self.modulate2d(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.modulate2d(self.norm2(x), shift_mlp, scale_mlp))
        return [x, text]

class Text_Guide_Image_0627(nn.Module):
    def __init__(self, dim, hidden_size, num_heads, bias):
        super().__init__()
        self.image_proj = nn.Linear(dim, hidden_size)
        self.text_to_kv_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_image_query = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, bias=bias, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, dim)

    def forward(self, image_feature, text_feature):
        B, C, H, W = image_feature.size()
        image_seq = image_feature.view(B, C, H * W).permute(0, 2, 1)
        image_query = self.image_proj(image_seq)
        text_kv = self.text_to_kv_proj(text_feature)
        output, _ = self.cross_attn_image_query(image_query, text_kv, text_kv)
        output_image = self.output_proj(output).permute(0, 2, 1).view(B, C, H, W)
        return image_feature + output_image

class UnetTransformer_with_ca_0627(nn.Module):
    def __init__(self, dim, num_blocks, heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(UnetTransformer_with_ca_0627, self).__init__()
        
        # Encoder 1
        self.encoder_level1_cross = nn.Sequential(*[TransformerDiT_0627(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_self = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2_img = Downsample(dim)
        self.down1_2_text = Downsample(dim)

        # Encoder 2
        self.encoder_level2_cross = nn.Sequential(*[TransformerDiT_0627(dim=int(dim*2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level2_self = nn.Sequential(*[TransformerBlock(dim=int(dim*2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3_img = Downsample(int(dim*2))
        self.down2_3_text = Downsample(int(dim*2))

        # Encoder 3
        self.encoder_level3_cross = nn.Sequential(*[TransformerDiT_0627(dim=int(dim*4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_level3_self = nn.Sequential(*[TransformerBlock(dim=int(dim*4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        # Decoders
        self.up3_2 = Upsample(int(dim*4))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*4), int(dim*2), kernel_size=1, bias=bias)
        self.decoder_level2_self = nn.Sequential(*[TransformerBlock(dim=int(dim*2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up2_1 = Upsample(int(dim*2))
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.decoder_level1_self = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

    def forward(self, inp_img, text_feature_global):
        # L1
        out_enc_level1 = self.encoder_level1_cross([inp_img, text_feature_global])
        out_enc_level1[0] = self.encoder_level1_self(out_enc_level1[0])
        
        # Down L1->L2
        inp_enc_level2 = self.down1_2_img(out_enc_level1[0])
        text_level2 = self.down1_2_text(text_feature_global)
        
        # L2
        out_enc_level2 = self.encoder_level2_cross([inp_enc_level2, text_level2])
        out_enc_level2[0] = self.encoder_level2_self(out_enc_level2[0])
        
        # Down L2->L3
        inp_enc_level3 = self.down2_3_img(out_enc_level2[0])
        text_level3 = self.down2_3_text(text_level2)
        
        # L3
        out_enc_level3 = self.encoder_level3_cross([inp_enc_level3, text_level3])
        out_enc_level3[0] = self.encoder_level3_self(out_enc_level3[0])
        
        # Decode L3->L2
        inp_dec_level2 = self.up3_2(out_enc_level3[0])
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2[0]], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2_self(inp_dec_level2)

        # Decode L2->L1
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1[0]], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        
        return self.decoder_level1_self(inp_dec_level1)

# --- Main Model ---

class TMGformer(nn.Module):
    def __init__(self,
        inp_channels=9, 
        out_channels=8,
        dim = 32,
        num_blocks = [4,3,3,2],
        num_refinement_blocks = 1,
        head  = 8,
        heads = [1,2,4,8],
        ffn_expansion_factor = 1,
        ffn_factor = 2.0,
        bias = False,
        text_dim = 768,
        LayerNorm_type = 'WithBias',
        channels_to_select = [0, 1, 2],
        inp_channels_1 = 3
    ):
        super(TMGformer, self).__init__()
        
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed_ms = OverlapPatchEmbed(inp_channels_1, dim)
        self.channels_to_select = channels_to_select
        
        # Text encoding branch
        self.TVFR = TransformerDiT(dim=dim, hidden_size=text_dim, num_heads=head,
                                            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        
        # Pre-processing block
        self.restormer1 = TransformerBlock(dim, head, ffn_factor, bias, LayerNorm_type)
        self.prelu1 = nn.PReLU()
        
        # Main U-Net
        self.AMF = UnetTransformer_with_ca_0627(
                            dim = dim,
                            num_blocks = num_blocks, 
                            heads = heads,
                            ffn_expansion_factor = ffn_expansion_factor,
                            bias = bias,
                            LayerNorm_type = LayerNorm_type)
                            
        self.output = nn.Sequential(nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias))

    def forward(self, pan, ms, text_ms):
        # Upsample MS
        ms = F.interpolate(ms, scale_factor=4, mode='bilinear', align_corners=False)
        
        # 1. Process MS Branch with Text
        selected_ms = ms[:, self.channels_to_select, :, :] # Select RGB
        ms_feature = self.patch_embed_ms(selected_ms)
        text_feature = self.TVFR([ms_feature, text_ms])
        
        # 2. Process Main Image Branch (PAN + MS)
        input_image = torch.cat([pan, ms], dim=1)
        img_feature = self.restormer1(self.prelu1(self.patch_embed(input_image)))
        
        # 3. Fuse in UNet
        # Note: self.unet takes (img, text_feature_global)
        # text_feature returns [x, text]. We use the transformed image feature (text_feature[0]) as the guidance? 
        # Checking original code logic: img_feature = self.unet(img_feature, text_feature[0])
        img_feature = self.AMF(img_feature, text_feature[0])
        
        # 4. Output
        img_feature = self.output(img_feature)
        img_feature = img_feature + ms # Residual connection
        return img_feature
