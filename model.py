import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as T
from transformers import AutoImageProcessor, AutoModel
from DEIMv2.tools.inference import onnx_inf

DEIM_MODEL_PATH = "model/best_stg2.onnx"
DINO_MODEL_PATH = "image_feature_extractor"

def load_detector(providers: list[str] | None = None) -> ort.InferenceSession:
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ort.InferenceSession(DEIM_MODEL_PATH, providers=providers)

def load_embedder(device: str | torch.device | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_PATH)
    model = AutoModel.from_pretrained(DINO_MODEL_PATH, trust_remote_code=True).to(device).eval()

    return processor, model, device

def run_detector(
    detector: ort.InferenceSession,
    image_pil: Image.Image,
    model_size: str = 's',
    score_thr: float = 0.4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # Dynamically extract expected input size (e.g., 640)
    size = detector.get_inputs()[0].shape[2]

    # Preprocess using DEIMv2's native utility
    resized_im_pil, ratio, pad_w, pad_h = onnx_inf.resize_with_aspect_ratio(image_pil, size)
    orig_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if model_size not in ['atto', 'femto', 'pico', 'n']
        else T.Lambda(lambda x: x)
    ])
    
    im_data = transforms(resized_im_pil).unsqueeze(0)

    # Run inference
    output = detector.run(
        output_names=None,
        input_feed={'images': im_data.numpy(), "orig_target_sizes": orig_size.numpy()}
    )

    labels, boxes, scores = output

    # Extract first batch item
    lab = labels[0]
    box = boxes[0]
    scr = scores[0]

    # Apply threshold to prevent downstream bottlenecks
    mask = scr > score_thr
    lab = lab[mask]
    box = box[mask]
    scr = scr[mask]

    # Revert padding and scaling logic to match the original image coordinates
    adjusted_boxes = np.zeros_like(box)
    if len(box) > 0:
        adjusted_boxes[:, 0] = (box[:, 0] - pad_w) / ratio
        adjusted_boxes[:, 1] = (box[:, 1] - pad_h) / ratio
        adjusted_boxes[:, 2] = (box[:, 2] - pad_w) / ratio
        adjusted_boxes[:, 3] = (box[:, 3] - pad_h) / ratio

    return adjusted_boxes, scr, lab

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