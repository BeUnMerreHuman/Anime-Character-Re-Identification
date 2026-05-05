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

### 3. Clone the anime character detector (dependency)

```bash
git clone https://github.com/ksasao/anime-character-detector.git anime_character_detector
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