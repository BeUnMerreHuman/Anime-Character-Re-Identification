import os
import yaml
import torch
import faiss
import pickle
import numpy as np
import torchvision.transforms as T
import torch.nn as nn
from pathlib import Path
from PIL import Image

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


def load_pipeline(config_path="models\\config.json", weights_path="models\\best_stg2.pth"):
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
        
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    
    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]["module"] if "module" in checkpoint["ema"] else checkpoint["ema"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
        
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    print("[model] Model ready.")
    
    return model, device


class PersonDatabase:
    def __init__(self, db_path="reid_db.pkl", threshold=0.4, variance_warning=0.3):
        self.db_path          = db_path
        self.threshold        = threshold
        self.variance_warning = variance_warning
        self.identities       = {}
        self.faiss_index      = None
        self.idx_to_name      = {}
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.identities = pickle.load(f)
        self.rebuild_index()

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.identities, f)

    def rebuild_index(self):
        if not self.identities:
            self.faiss_index = None
            self.idx_to_name = {}
            return

        first_name = next(iter(self.identities))
        dim = self.identities[first_name][0].shape[0]

        self.faiss_index = faiss.IndexFlatL2(dim)
        self.idx_to_name = {}

        for idx, (name, embs) in enumerate(self.identities.items()):
            mean_emb = np.mean(embs, axis=0)
            centroid = mean_emb / np.linalg.norm(mean_emb)
            self.faiss_index.add(np.array([centroid], dtype=np.float32))
            self.idx_to_name[idx] = name

    def search(self, emb):
        if self.faiss_index is None:
            return None, None
        distances, indices = self.faiss_index.search(
            np.array([emb], dtype=np.float32), k=1
        )
        dist = distances[0][0]
        if dist < self.threshold:
            return self.idx_to_name[indices[0][0]], dist
        return None, dist

    def add_embedding(self, name: str, emb: np.ndarray):
        if name not in self.identities:
            self.identities[name] = []
        self.identities[name].append(emb)
        self.save()
        self.rebuild_index()

    def get_names(self):
        return list(self.identities.keys())


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