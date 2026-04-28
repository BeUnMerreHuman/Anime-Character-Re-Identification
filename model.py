import os
import yaml 
import torch
import numpy as np
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from DEIMv2.engine.backbone import DINOv3STAs
from DEIMv2.engine.deim import HybridEncoder, DEIMTransformer
from DEIMv2.engine.deim.postprocessor import PostProcessor

class DEIMv2_Local(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.backbone = DINOv3STAs(**config.get("DINOv3STAs", {}))
        self.encoder  = HybridEncoder(**config.get("HybridEncoder", {}))
        self.decoder = DEIMTransformer(**config.get("DEIMTransformer", {}))
        self.postprocessor = PostProcessor(**config.get("PostProcessor", {}))
        
        self._captured_embeddings = []
        target_layer = self.decoder.decoder.layers[-1]
        target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self._captured_embeddings.append(output.detach().cpu())

    def forward(self, x, orig_target_sizes):
        self._captured_embeddings.clear()

        x_feat = self.backbone(x)
        x_enc  = self.encoder(x_feat)
        x_dec  = self.decoder(x_enc)
        detections = self.postprocessor(x_dec, orig_target_sizes)
        
        latent_dna = self._captured_embeddings[0]

        return {
            "detections": detections,
            "embeddings": latent_dna,
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
    T.Resize((640, 640)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_image(image: Image.Image, model, device, threshold: float = 0.50):
    image = image.convert("RGB") 
    w, h = image.size
    orig_size = torch.tensor([[w, h]], dtype=torch.long, device=device)
    tensor    = _TRANSFORMS(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor, orig_size)

    detections = outputs["detections"]
    embeddings = outputs["embeddings"]
    raw_dec    = outputs["raw_dec"]

    det_result = detections[0] if isinstance(detections, (list, tuple)) else detections
    
    labels = det_result["labels"][0] if det_result["labels"].dim() > 1 else det_result["labels"]
    boxes  = det_result["boxes"][0] if det_result["boxes"].dim() > 2 else det_result["boxes"]
    scores = det_result["scores"][0] if det_result["scores"].dim() > 1 else det_result["scores"]

    valid = (scores > threshold)
    valid_boxes  = boxes[valid].cpu().numpy()
    valid_scores = scores[valid].cpu()

    logits = raw_dec["pred_logits"][0]
    prob = logits.sigmoid()
    class_0_probs = prob[:, 0].cpu()

    valid_embeddings = []
    for score in valid_scores:
        best_idx = torch.argmin(torch.abs(class_0_probs - score))
        valid_embeddings.append(embeddings[0][best_idx].numpy())

    valid_embeddings = np.array(valid_embeddings) if valid_embeddings else np.empty((0, embeddings.shape[-1]))

    return valid_boxes, valid_scores.numpy(), valid_embeddings