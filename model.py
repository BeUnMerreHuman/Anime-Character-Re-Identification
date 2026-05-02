import os
import yaml 
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.ops as ops
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from DEIMv2.engine.backbone import DINOv3STAs
from DEIMv2.engine.deim import HybridEncoder, DEIMTransformer
from DEIMv2.engine.deim.postprocessor import PostProcessor

class Letterbox:
    def __init__(self, target_size=640, fill=0):
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
        
        self._patch_tokens = None
        
        # Register a forward hook to intercept tokens in a single pass
        def hook_fn(module, input, output):
            # Output shape: [B, Seq_Len, Embed_Dim]
            # DINOv3 prepends 1 CLS token and a dynamic number of storage tokens.
            n_skip = self.backbone.dinov3.n_storage_tokens + 1
            self._patch_tokens = output[:, n_skip:, :]
            
        self.backbone.dinov3.norm.register_forward_hook(hook_fn)

    def forward(self, x, orig_target_sizes):
        # Reset tokens for the current pass
        self._patch_tokens = None
        
        # 1. Single forward pass. This executes the backbone and triggers the hook.
        x_feat = self.backbone(x)
        
        # 2. Retrieve intercepted patch tokens securely
        if self._patch_tokens is None:
            raise RuntimeError("Forward hook failed to capture patch tokens. Backbone execution bypassed the norm layer.")
        
        patch_tokens = self._patch_tokens

        x_enc  = self.encoder(x_feat)
        x_dec  = self.decoder(x_enc)
        detections = self.postprocessor(x_dec, orig_target_sizes)
        
        return {
            "detections": detections,
            "backbone_features": x_feat,
            "patch_tokens": patch_tokens, # Exposing the intercepted pure 1D semantic tokens
            "raw_dec": x_dec  
        }


def load_pipeline(config_path="model/config.json", weights_path="model/model.safetensors"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] Using device: {device}")

    print(f"[model] Loading config from {config_path}...")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("[model] Building architecture...")
    model = DEIMv2_Local(config)
    
    print(f"[model] Loading weights from {weights_path}...")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file missing: {weights_path}")
        
    state_dict = load_file(weights_path)

    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    print("[model] Model ready.")
    
    return model, device


_TRANSFORMS = T.Compose([
    Letterbox(640),
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

    orig_size = torch.tensor([[w, h]], dtype=torch.long, device=device)
    # The Letterbox transform strictly guarantees a 640x640 tensor
    tensor = _TRANSFORMS(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor, orig_size)

    raw_dec = outputs["raw_dec"]
    patch_tokens = outputs["patch_tokens"] # Shape: (1, N, 384)
    
    # 1. Reshape the 1D ViT sequence back into a 2D spatial map
    B, N, C = patch_tokens.shape
    # Patch size is 16. For a 640x640 input, H and W are exactly 40.
    H_feat = W_feat = 640 // 16 
    
    # Target shape for RoI Align: (B, C, H, W)
    spatial_map = patch_tokens.view(B, H_feat, W_feat, C).permute(0, 3, 1, 2).contiguous()

    logits = raw_dec["pred_logits"][0]    
    raw_boxes = raw_dec["pred_boxes"][0]   

    probs = logits.sigmoid()
    class_0_probs = probs[:, 0]
    valid_mask = class_0_probs > threshold

    valid_scores = class_0_probs[valid_mask].cpu().numpy()
    valid_raw_boxes = raw_boxes[valid_mask]

    if len(valid_scores) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0, C))

    # Box translation to canvas-scale coordinates
    cx_canvas, cy_canvas, bw_canvas, bh_canvas = valid_raw_boxes.unbind(-1)
    
    cx_px = cx_canvas * 640.0
    cy_px = cy_canvas * 640.0
    bw_px = bw_canvas * 640.0
    bh_px = bh_canvas * 640.0
    
    cx_orig = (cx_px - pad_left) / scale
    cy_orig = (cy_px - pad_top) / scale
    w_orig = bw_px / scale
    h_orig = bh_px / scale
    
    x1_orig = cx_orig - 0.5 * w_orig
    y1_orig = cy_orig - 0.5 * h_orig
    x2_orig = cx_orig + 0.5 * w_orig
    y2_orig = cy_orig + 0.5 * h_orig
    
    valid_boxes_orig = torch.stack([x1_orig, y1_orig, x2_orig, y2_orig], dim=-1).cpu().numpy()

    # Formulate absolute RoI boxes
    x1_roi = cx_px - 0.5 * bw_px
    y1_roi = cy_px - 0.5 * bh_px
    x2_roi = cx_px + 0.5 * bw_px
    y2_roi = cy_px + 0.5 * bh_px
    
    batch_idx = torch.zeros_like(x1_roi)
    roi_boxes = torch.stack([batch_idx, x1_roi, y1_roi, x2_roi, y2_roi], dim=1)

    # 2. Strict spatial scale configuration
    spatial_scale = 1.0 / 16.0 

    # 3. Execute RoI Align with a high-density 14x14 grid
    roi_features = ops.roi_align(
        spatial_map,
        boxes=roi_boxes,
        output_size=(14, 14), 
        spatial_scale=spatial_scale,
        aligned=True
    )

    # 4. Mean Pooling (Global Average Pooling) 
    # Replaces 'amax' to properly preserve the average structural representation
    pooled_features = roi_features.mean(dim=[2, 3])
    
    # L2 Normalize
    valid_embeddings = F.normalize(pooled_features, p=2, dim=1).cpu().numpy()

    return valid_boxes_orig, valid_scores, valid_embeddings