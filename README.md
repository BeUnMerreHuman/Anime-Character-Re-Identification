## 🚀 Quick Start

Follow these steps to set up the environment and run the application:

### 1. Clone the Repository
```bash
git clone https://github.com/BeUnMerreHuman/Anime-Character-Re-Identification.git
cd Anime-Character-Re-Identification
```

### 2. Install Dependencies
This project uses `uv` for fast, reproducible dependency management.
```bash
uv sync
```

### 3. Download Model Weights
Download the required weights from Hugging Face into the local directory:
```bash
hf download Be-Un-Merre-Human/Anime_Character_Detector --local-dir ./model
```

### 4. Launch the App
```bash
uv run app.py
```
