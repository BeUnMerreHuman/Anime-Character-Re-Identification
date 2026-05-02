import os
import yaml 
import torch
import numpy as np
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from DEIMv2.engine.backbone import DINOv3STAs
from DEIMv2.engine.deim import HybridEncoder, DEIMTransformer
from DEIMv2.engine.deim.postprocessor import PostProcessor

class Letterbox:
    def __init__(self, target_size=640, fill=128):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img: Image.Image):
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img = T.functional.resize(img, (new_h, new_w))
        
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
        
        return T.functional.pad(img, padding, fill=self.fill)

class DEIMv2_Local(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = DINOv3STAs(**config.get("DINOv3STAs", {}))
        self.encoder  = HybridEncoder(**config.get("HybridEncoder", {}))
        self.decoder = DEIMTransformer(**config.get("DEIMTransformer", {}))
        self.postprocessor = PostProcessor(**config.get("PostProcessor", {}))

    def forward_detection(self, x):
        """Pass 1: Full pass for detection."""
        x_feat = self.backbone(x)
        x_enc  = self.encoder(x_feat)
        x_dec  = self.decoder(x_enc)
        return x_dec

    def extract_tokens(self, x):
        features = self.backbone.dinov3.forward_features(x)
        cls_token = features["x_norm_clstoken"]       
        patch_tokens = features["x_norm_patchtokens"] 
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
    

def load_pipeline(config_path="model/config.json", weights_path="model/model.safetensors"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] Using device: {device}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = DEIMv2_Local(config)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file missing: {weights_path}")
        
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    
    return model, device


# Separate pipelines for detection and recognition
_TRANSFORMS_DET = T.Compose([
    Letterbox(640),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_TRANSFORMS_REC = T.Compose([
    Letterbox(384),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def process_image(image: Image.Image, model, device, threshold: float = 0.50):
    image = image.convert("RGB") 
    w, h = image.size
    
    scale = 640 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_left = (640 - new_w) // 2
    pad_top = (640 - new_h) // 2

    # --- PASS 1: Detection Area ---
    tensor_det = _TRANSFORMS_DET(image).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_dec = model.forward_detection(tensor_det)

    logits = raw_dec["pred_logits"][0]    
    raw_boxes = raw_dec["pred_boxes"][0]   

    probs = logits.sigmoid()
    class_0_probs = probs[:, 0]
    valid_mask = class_0_probs > threshold

    valid_scores = class_0_probs[valid_mask].cpu().numpy()
    valid_raw_boxes = raw_boxes[valid_mask]

    if len(valid_scores) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0, 768)) # 768 embed_dim fallback

    # Box translation to canvas-scale coordinates
    cx_canvas, cy_canvas, bw_canvas, bh_canvas = valid_raw_boxes.unbind(-1)
    
    cx_px, cy_px = cx_canvas * 640.0, cy_canvas * 640.0
    bw_px, bh_px = bw_canvas * 640.0, bh_canvas * 640.0
    
    cx_orig, cy_orig = (cx_px - pad_left) / scale, (cy_px - pad_top) / scale
    w_orig, h_orig = bw_px / scale, bh_px / scale
    
    x1_orig, y1_orig = cx_orig - 0.5 * w_orig, cy_orig - 0.5 * h_orig
    x2_orig, y2_orig = cx_orig + 0.5 * w_orig, cy_orig + 0.5 * h_orig
    
    valid_boxes_orig = torch.stack([x1_orig, y1_orig, x2_orig, y2_orig], dim=-1).cpu().numpy()

    # --- PASS 2: Recognition (Cropping and CLS token extraction) ---
    crop_tensors = []
    valid_indices = []

    for idx, box in enumerate(valid_boxes_orig):
        x1, y1, x2, y2 = box
        # Clip to ensure valid PIL crop coordinates
        x1_c, y1_c = max(0, x1), max(0, y1)
        x2_c, y2_c = min(w, x2), min(h, y2)
        
        if x2_c <= x1_c or y2_c <= y1_c:
            continue
            
        crop = image.crop((x1_c, y1_c, x2_c, y2_c))
        # Zero-pad letterbox to 224x224
        crop_tensors.append(_TRANSFORMS_REC(crop))
        valid_indices.append(idx)

    # Filter out any invalid boxes that had area <= 0
    valid_boxes_orig = valid_boxes_orig[valid_indices]
    valid_scores = valid_scores[valid_indices]

    if len(crop_tensors) == 0:
         return valid_boxes_orig, valid_scores, np.empty((0, 768))

    # Batch crops for speed
    batch_crops = torch.stack(crop_tensors).to(device)

    with torch.no_grad():
        # Extracted directly from DINOv3 core without adapter execution
        cls_tokens = model.extract_tokens(batch_crops)
        valid_embeddings = F.normalize(cls_tokens, p=2, dim=1).cpu().numpy()

    return valid_boxes_orig, valid_scores, valid_embeddings