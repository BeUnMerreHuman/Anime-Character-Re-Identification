# 🚀 Anime Character Re-Identification

This repository contains an end-to-end, zero-shot anime character detection and re-identification system designed to locate and identify characters across images and video sequences without requiring model retraining for new characters.

By replacing traditional, fixed-class classification systems (like standard YOLO models) with an embedding-based similarity search pipeline, the system can dynamically scale to handle expanding character rosters across large franchises.

---

## ✨ Key Features

* **Zero-Shot Recognition:** Identifies previously seen anime characters using high-dimensional embedding similarity instead of fixed class layers.


* **Domain-Adapted Models:** Features foundation vision models explicitly fine-tuned to overcome the challenges of anime-style artwork, visual proportions, and shading.


* **Vector Database Backend:** Utilizes LanceDB to manage, store, and query character identity embeddings dynamically over time.


* **ONNX Deployment:** Both detection and embedding extraction models are compiled to ONNX format to maximize portable and efficient cross-platform inference.



---

## 🛠️ System Architecture & Methodology

The system processes video frames or standalone images through a multi-stage pipeline:

1. **Character Detection (DEIMv2):** Every frame is processed by a fine-tuned DEIMv2-Large model (32.2M parameters) to detect and output character bounding boxes.


2. **Feature Extraction (DINOv3):** Detected regions are cropped and passed through a DINOv3 ViT-B model (86M parameters) adapted via Low-Rank Adaptation (LoRA) to generate unique character embeddings.


3. **Vector Database Querying (LanceDB):** The generated embeddings are evaluated against an existing identity index in LanceDB using a dot-product metric.


4. **Identity Management:** The system handles identity logic based on similarity thresholds:


* **Confident Match ($\geq 0.80$):** Automatically assigns the existing character identity.


* **Uncertain Match ($0.69$ to $0.79$):** Resolves identities using a top-k similarity voting mechanism combined with observation counts.


* **New Identity ($< 0.69$):** Registers a new identity cluster and saves a baseline thumbnail.





---

## 🚀 Quick Start

Follow these steps to set up the environment and run the application:

### Setup Instructions

#### 1. Clone this repository

```bash
git clone https://github.com/BeUnMerreHuman/Anime-Character-Re-Identification.git
cd Anime-Character-Re-Identification

```

#### 2. Install dependencies

```bash
uv sync

```

#### 3. Download detection model

The fine-tuned DEIMv2 model is hosted as a Notebook Output on Kaggle:

```bash
kaggle models instances versions download muneeburrehman98/deimv2-anime-character-detector/onnx/default/1

```

#### 4. Download recognition model

The LoRA-adapted DINOv3 model is hosted as a Notebook Output on Kaggle:

```bash
kaggle kernels output muneeburrehman98/dinov3-finetune-anime -p DINOv3 --force

```

### ▶️ Run the Application

```bash
uv run app.py

```

---

## 📊 Performance & Benchmarks

* **GPU Inference (Google Colab T4):** Achieves approximately 1.5 seconds of execution computation time per second of processed video.


* **CPU Inference (Core i3 3rd Gen):** Supports standard image inference at approximately 12 seconds of computation time per single image.


* **Dataset Foundation:** Models were fine-tuned using the Danbooru annotated dataset mapping character localized bounding boxes.



