import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from anime_character_detector.python.onnx_predictor import YoloxOnnxPredictor

YOLOX_MODEL_PATH = "anime_character_detector/python/character.onnx"
YOLOX_INPUT_SHAPE = (640, 640)
YOLOX_CLASS_NAMES = ["character"]
DINO_MODEL_PATH = "image_feature_extractor"

def load_detector(
    score_thr: float = 0.3,
    nms_thr: float = 0.45,
    providers: list[str] | None = None,
) -> YoloxOnnxPredictor:
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return YoloxOnnxPredictor(
        model_path=YOLOX_MODEL_PATH,
        input_shape=YOLOX_INPUT_SHAPE,
        score_thr=score_thr,
        nms_thr=nms_thr,
        class_names=YOLOX_CLASS_NAMES,
        providers=providers,
    )

def load_embedder(device: str | torch.device | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_PATH)
    model = AutoModel.from_pretrained(DINO_MODEL_PATH, trust_remote_code=True).to(device).eval()

    return processor, model, device

def run_detector(
    detector: YoloxOnnxPredictor,
    image_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes, scores, cls_ids = detector.predict(image_np)
    return boxes, scores, cls_ids

def run_embedder(
    processor,
    model,
    device: str | torch.device,
    crop: Image.Image,
) -> np.ndarray:
    inputs = processor(images=crop.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    cls_token = outputs.last_hidden_state[:, 0, :]
    emb = cls_token.squeeze(0).cpu().numpy().astype(np.float32)

    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb


def load_pipeline():
    detector = load_detector()
    processor, model, device = load_embedder()
    return detector, processor, model, device