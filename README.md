# 🚀 Quick Start

Follow these steps to set up the environment and run the application:

## ⚠️ Requirements

* You **must have a Hugging Face account**
* You **must accept the model license agreement** on the model page:
  https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
* You **must be logged in via CLI with a read token**
* Paste your **Hugging Face Read Token** when prompted.

## 🚀 Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/BeUnMerreHuman/Anime-Character-Re-Identification.git
cd Anime-Character-Re-Identification
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Download detection model

Model is already uploaded

```bash
kaggle kernels output muneeburrehman98/deimv2-s-finetune -p model
```

### 4. Download the DINOv3 model (requires access approval)

```bash
hf download facebook/dinov3-vits16-pretrain-lvd1689m --local-dir ./image_feature_extractor
```

> If this step fails, you likely have **not accepted the model terms** or are **not authenticated**.

---

## ▶️ Run the Application

```bash
uv run app.py
```