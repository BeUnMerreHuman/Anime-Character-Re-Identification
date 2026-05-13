# 🚀 Quick Start

Follow these steps to set up the environment and run the application:

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

Model is saved as Notebooks Output

```bash
kaggle kernels output muneeburrehman98/deimv2-finetune-anime -p DEIMv2 --force
```

### 4. Download recognition model

Model is saved as Notebooks Output

```bash
kaggle kernels output muneeburrehman98/dinov3-finetune-anime -p DINOv3 --force
```

## ▶️ Run the Application

```bash
uv run app.py
```