import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import h5py
import os
import kornia
import clip
import numpy as np

from config import Config
from model import TMGformer
from clip_utils import Predictor, extract_clip_feature_differentiable
from loss import SSIMLoss, cosine_similarity_loss_normalized, InfoNCELossWithGT
from utils import DifferentiableDynamicVisualizer, DifferentiableCLIPPreprocess, write_list_to_txt
from metricsm import get_metrics_reduced

# --- Setup Environment ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = Config.device
print(f"Using device: {device}")

# --- Load Data ---
def load_dataset(data_path, text_path, max_val=Config.max_val):
    print(f"Loading data from {data_path}...")
    with h5py.File(data_path, 'r') as f, h5py.File(text_path, 'r') as f_text:
        hrms = np.array(f['gt']) / max_val
        lrms = np.array(f['ms']) / max_val
        pan = np.array(f['pan']) / max_val
        text_ms = np.array(f_text['text'])
    
    # Convert to Tensor
    t_pan = torch.FloatTensor(pan)
    t_lrms = torch.FloatTensor(lrms)
    t_hrms = torch.FloatTensor(hrms)
    t_text = torch.FloatTensor(text_ms)
    
    return TensorDataset(t_pan, t_lrms, t_text, t_hrms)

train_dataset = load_dataset(Config.train_data_path, Config.train_text_path)
train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)

test_dataset = load_dataset(Config.test_data_path, Config.test_text_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Train batches: {len(train_loader)}")

# --- Initialize Model ---
model = TMGformer(
    inp_channels=Config.inp_channels,
    out_channels=Config.out_channels,
    dim=Config.dim,
    text_dim=Config.text_dim,
    channels_to_select=Config.channels_to_select
).to(device)

optimizer = Adam(model.parameters(), lr=Config.lr)

# --- Initialize Visualizers & Loss Components ---
dynamic_visualizer = DifferentiableDynamicVisualizer(
    num_input_channels=Config.inp_channels, 
    channels_to_select=Config.channels_to_select, 
    output_order=[2, 1, 0]
).to(device)

diff_preprocess = DifferentiableCLIPPreprocess().to(device)
predictor = Predictor(device=device)

# Load CLIP model for feature extraction
clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
clip_model.eval()

# Losses
loss_l1 = torch.nn.L1Loss()
loss_ssim = SSIMLoss()
loss_infonce = InfoNCELossWithGT().to(device)
loss_clip_text = cosine_similarity_loss_normalized

# --- 5. Main Training Loop ---
for epoch in range(Config.start_epoch, Config.epochs):
    
    # === Training Phase ===
    model.train()
    print(f"\nTraining Epoch: {epoch}")
    
    for index, (pan, lrms, lrms_text, hrms) in enumerate(train_loader):
        # Move to device
        pan = pan.to(device)
        lrms = lrms.to(device)
        lrms_text = lrms_text.to(device)
        hrms = hrms.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(pan, lrms, lrms_text)
        
        # 1. Pixel-wise Losses
        l1 = loss_l1(output, hrms)
        l_ssim = loss_ssim(output, hrms)
        # Gradient Loss (Edge preservation)
        l_grad = loss_l1(kornia.filters.SpatialGradient()(output), kornia.filters.SpatialGradient()(hrms))
        
        # 2. Semantic/CLIP Losses
        # Differentiable Visualization -> Preprocess -> Text Feature Extraction
        try:
            out_vis = dynamic_visualizer(output)
            hrms_vis = dynamic_visualizer(hrms)
            
            out_processed = diff_preprocess(out_vis)
            hrms_processed = diff_preprocess(hrms_vis)
            
            # Text-Image Consistency Loss
            out_feat_text = predictor.predict_feature(out_processed)
            hrms_feat_text = predictor.predict_feature(hrms_processed)
            l_clip = loss_clip_text(out_feat_text, hrms_feat_text, pool_type="mean")
        except Exception as e:
            print(f"CLIP Loss Warning: {e}")
            l_clip = torch.tensor(0.0).to(device)
        
        # Total Weighted Loss
        loss_total = (Config.weight_l1 * l1 + 
                      Config.weight_clip * l_clip + 
                      Config.weight_ssim * l_ssim + 
                      Config.weight_grad * l_grad)
        
        # Backward & Step
        loss_total.backward()
        optimizer.step()
        
        if index % 10 == 0:
             print(f'Epoch: {epoch}/{Config.epochs} Batch: {index}/{len(train_loader)} '
                   f'L1: {l1.item():.4f} CLIP: {l_clip.item():.4f} SSIM: {l_ssim.item():.4f} Total: {loss_total.item():.4f}')

    # === Evaluation Phase (Matching Original Logic) ===
    print('Evaluating Epoch:', epoch)
    model.eval()
    
    # Initialize metric lists
    psnr_loss, ssim, cc, sam, ergas = [], [], [], [], [] 
    
    for index, test_data in enumerate(test_loader):
        pan_batch, lrms_batch, lrms_t_batch, hrms_batch = test_data
        
        pan_batch    = pan_batch.to(device)
        lrms_batch   = lrms_batch.to(device)
        lrms_t_batch = lrms_t_batch.to(device)
        hrms_batch   = hrms_batch.to(device)
        
        with torch.no_grad():
            output = model(pan_batch, lrms_batch, lrms_t_batch)
            
            # --- Calculate Metrics ---
            try:
                # Assuming get_metrics_reduced returns 5 scalars: PSNR, SSIM, CC, SAM, ERGAS
                m1, m2, m3, m4, m5 = get_metrics_reduced(output, hrms_batch)
                psnr_loss.append(m1)
                ssim.append(m2)
                cc.append(m3)
                sam.append(m4)
                ergas.append(m5)
            except Exception as e:
                print(f"Warning: Error calculating metrics for batch {index}: {e}")

    # --- Calculate Mean Metrics & Save ---
    if psnr_loss:
        psnr_mean = np.mean(psnr_loss)
        ssim_mean = np.mean(ssim)
        cc_mean   = np.mean(cc)
        sam_mean  = np.mean(sam)
        ergas_mean= np.mean(ergas)

        # Print to console
        print(f"Eval Results - PSNR: {psnr_mean:.4f}, SSIM: {ssim_mean:.4f}, "
              f"CC: {cc_mean:.4f}, SAM: {sam_mean:.4f}, ERGAS: {ergas_mean:.4f}")

        # Save Weights
        os.makedirs(Config.save_dir, exist_ok=True)
        weight_save_path = os.path.join(Config.save_dir, f'{epoch}.pth')
        torch.save(model.state_dict(), weight_save_path)
        print(f"Model saved to {weight_save_path}")

        # Save Logs to txt file
        log_filepath = os.path.join(Config.save_dir, "eval_log.txt")
        # Creating a tuple/list entry for the log
        log_entry = [(epoch, psnr_mean, ssim_mean, cc_mean, sam_mean, ergas_mean)]
        write_list_to_txt(log_entry, log_filepath)

    else:
        print("Warning: No metrics collected during evaluation. Check test_loader or get_metrics_reduced function.")

    print('Finished Evaluation for Epoch:', epoch)

print('Finished Training')
