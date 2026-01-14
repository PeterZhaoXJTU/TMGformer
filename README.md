# [Insert Paper Title Here]

<div align="center">
  <!-- 请自行替换作者姓名和谷歌学术链接，如果没有链接可去掉 href 属性 -->
  <a href="[Link to Author 1]"> [Author Name 1] </a> |
  <a href="[Link to Author 2]"> [Author Name 2] </a> |
  <a href="[Link to Author 3]"> [Author Name 3] </a>

  <!-- 请替换单位名称 -->
  <br>
  <a>[Institution Name 1] </a>
</div>

<div align="center">
  <!-- 请替换论文 PDF 链接 -->
  [[Paper]([Link to Paper PDF])] 
  [[Code](https://github.com/[YourUsername]/TMGformer)]
  [[Data](https://drive.google.com/drive/folders/13QcDi_IxDgg7K5fa8VeAitO3u9HoEN8N?usp=drive_link)]
</div>

<br>

Official implementation of **TMGformer** for Pansharpening.

<!-- 请将结果对比图命名为 results.png 放入 assets 文件夹 -->
<div align="center">
<img src="assets/results.png" width="800" alt="Results Comparison">
<p><em>Figure 1: Visual results comparisons on Pansharpening datasets.</em></p>
</div>

<!-- 请将架构图命名为 architecture.png 放入 assets 文件夹 -->
<div align="center">
<img src="assets/architecture.png" width="800" alt="TMGformer Architecture">
<p><em>Figure 2: The overall architecture of the proposed TMGformer.</em></p>
</div>

# News
**[2026/01/14]**: The training and inference code of TMGformer is released.

# Fast Run

## 1. Environment Setup

Please follow the steps below to set up the environment:

```bash
# 1. Create a conda environment
conda create -n TMGformer python=3.9
conda activate TMGformer

# 2. Install PyTorch (CUDA 12.1)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Install other dependencies
pip install -r requirements.txt
```

## 2. Data Preparation

### Image Datasets (H5 Format)
We follow the **[Pan-Collection]** benchmark. Please download the H5 format datasets for **GF2, QB, and WV3**.
- **Download Link**: [[GitHub - PanCollection]](https://github.com/liangjiandeng/PanCollection)

### Text Features & Pre-trained Weights
The text feature datasets and necessary CLIP/GPT2 checkpoints are hosted on Google Drive.
- **Download Link**: [[Google Drive]](https://drive.google.com/drive/folders/13QcDi_IxDgg7K5fa8VeAitO3u9HoEN8N?usp=drive_link)

**Please download the following files from the Drive:**
1.  **Text Features**: `train_images_text_feature.h5` and `test_images_text_feature.h5` (for GF2, QB, and WV3).
2.  **Clipcap Weights**: `clip_weights_path` (e.g., `coco_prefix-009.pt`).
3.  **GPT2 Files**: `gpt2_tokenizer_path` and `gpt2_model_path`.

### Recommended Directory Structure
We recommend organizing your data as follows. **Important**: Please update the paths in `config.py` to match your local structure.

<details>
  <summary>Directory Structure (Click to unfold)</summary>
<pre><code>
TMGformer_github
├── assets/
│   ├── architecture.png
│   └── results.png
├── weight/                 # Training checkpoints will be saved here
├── data/
│   ├── PanCollection/      # Image datasets from PanCollection
│   │   ├── train_gf2.h5
│   │   ├── test_gf2.h5
│   │   └── ...
│   ├── text_features/      # From Google Drive
│   │   ├── train_images_text_feature.h5
│   │   └── test_images_text_feature.h5
│   └── pretrained/         # From Google Drive
│       ├── coco_prefix-009.pt
│       ├── tokenizer/
│       └── gpt2_model/
├── config.py               # <--- UPDATE PATHS HERE
├── train.py
├── requirements.txt
└── ...
</code></pre>
</details>

## 3. Training

To start training the TMGformer model, ensure your `config.py` paths are correct and run:

```bash
cd ./TMGformer_github
python train.py
```

# Results

You can access the trained models and visual results at [Link to your cloud drive if available].

# Citations

If you find this work useful, please kindly cite our paper:

```bibtex
@ARTICLE{TMGformer,
  author={[Author Names]},
  journal={[Journal Name]},
  title={TMGformer: [Full Paper Title]},
  year={2026},
  volume={},
  number={},
  pages={},
  doi={}
}
```
