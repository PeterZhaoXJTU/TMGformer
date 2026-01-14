import torch

class Config:
    # --- Hardware ---
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # --- Data Paths ---
    # Update these paths to your local directories
    # max_val = 1023 # GF2
    max_val = 2047 # QB/WV3
    train_data_path = "dataset/WV3/train_wv3.h5"
    train_text_path = "dataset/WV3/train_text_feature.h5"
    test_data_path = "dataset/WV3/test_wv3_multiExm1.h5"
    test_text_path = "dataset/WV3/reduce_test_text_feature.h5"
    
    # Weight Paths for CLIP/Captioning
    clip_weights_path = "clipcap/clipcap_weight.pt"
    gpt2_tokenizer_path = "clipcap/tokenizer"
    gpt2_model_path = "clipcap/model"

    # --- Save Paths ---
    save_dir = "weights/WV3"
    
    # --- Training Hyperparameters ---
    lr = 0.0005
    batch_size = 4
    epochs = 50
    start_epoch = 0
    
    # --- Model Hyperparameters ---
    # inp_channels = 5         # GF2/QB: PAN(1) + MS(4)
    # out_channels = 4
    dim = 32
    text_dim = 768
    # channels_to_select = [0, 1, 2] # RGB indices for QB/GF2

    inp_channels = 9         # WV3: PAN(1) + MS(8)
    out_channels = 8
    channels_to_select = [0, 2, 4] # RGB indices for WV3

    # --- Loss Weights ---
    weight_l1 = 1.0
    weight_clip = 0.001
    weight_ssim = 0.01
    weight_grad = 0.1
