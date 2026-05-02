import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Import your existing pipeline
from model import load_pipeline, process_image, _TRANSFORMS_REC

# Global dictionary to store the intercepted attention weights
_ATTENTION_STORE = {}

def get_attention_hook(name):
    """PyTorch forward hook to intercept the QKV output."""
    def hook(module, input, output):
        _ATTENTION_STORE[name] = output
    return hook

def extract_attention_via_hook(model, image_tensor):
    """
    Forces the extraction of attention weights from the custom DinoVisionTransformer
    by hooking into the last SelfAttentionBlock's qkv linear layer.
    """
    device = image_tensor.device
    last_block = model.backbone.dinov3.blocks[-1]
    
    qkv_layer = None
    for name, module in last_block.named_modules():
        if 'qkv' in name:
            qkv_layer = module
            break
            
    if qkv_layer is None:
        raise AttributeError("Could not find a 'qkv' layer in the final SelfAttentionBlock.")

    # Register the hook
    handle = qkv_layer.register_forward_hook(get_attention_hook('last_qkv'))
    
    # Run the forward pass
    with torch.no_grad():
        model.extract_tokens(image_tensor)
        
    # Remove the hook
    handle.remove()
    
    # Reconstruct the attention map
    qkv_output = _ATTENTION_STORE['last_qkv'] 
    B, N, C = qkv_output.shape
    embed_dim = C // 3
    num_heads = model.backbone.dinov3.num_heads 
    head_dim = embed_dim // num_heads
    
    qkv = qkv_output.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)
    
    # --- THE FIX: Account for DINOv3 Storage Tokens ---
    # Layout: [CLS] (0) | [Storage] (1 to n_storage_tokens) | [Patches] (n_storage_tokens+1 to end)
    n_storage_tokens = model.backbone.dinov3.n_storage_tokens
    patch_start_idx = 1 + n_storage_tokens
    
    # Extract [CLS] token's attention specifically to the patch tokens (ignoring storage tokens)
    cls_attn = attn[:, :, 0, patch_start_idx:]
    
    # Reshape to grid and average across heads
    grid_size = int(cls_attn.shape[-1] ** 0.5)
    cls_attn = cls_attn.reshape(B, num_heads, grid_size, grid_size)
    attn_map = cls_attn.mean(dim=1)[0].cpu().numpy() 
    
    return attn_map

def generate_diagnostics(image_path, config_path="model/config.json", weights_path="model/model.safetensors"):
    print(f"Loading pipeline and processing {image_path}...")
    model, device = load_pipeline(config_path, weights_path)
    
    original_img = Image.open(image_path).convert("RGB")
    boxes, scores, _ = process_image(original_img, model, device, threshold=0.50)
    
    if len(boxes) == 0:
        print("No detections found. Aborting.")
        return

    w, h = original_img.size
    
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        crop = original_img.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
        tensor_crop = _TRANSFORMS_REC(crop).unsqueeze(0).to(device)
        
        # Reverse normalization for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_view = tensor_crop[0].cpu() * std + mean
        img_view = torch.clamp(img_view, 0, 1).permute(1, 2, 0).numpy()
        img_view = (img_view * 255).astype(np.uint8)
        
        # Execute Hook and extract
        attn_map = extract_attention_via_hook(model, tensor_crop)
        
        # Resize and map to colors
        attn_map_resized = cv2.resize(attn_map, (img_view.shape[1], img_view.shape[0]))
        attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min() + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(img_view, 0.5, heatmap, 0.5, 0)
        
        # Save to disk directly
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img_view)
        axs[0].set_title(f"Crop {idx} (Score: {scores[idx]:.2f})")
        axs[0].axis('off')
        
        axs[1].imshow(attn_map_resized, cmap='jet')
        axs[1].set_title("DINOv3 [CLS] Attention")
        axs[1].axis('off')
        
        axs[2].imshow(overlay)
        axs[2].set_title("Overlay")
        axs[2].axis('off')
        
        out_name = f"diagnostic_crop_{idx}.png"
        plt.tight_layout()
        plt.savefig(out_name)
        plt.close()
        print(f"Saved diagnostic -> {out_name}")

if __name__ == "__main__":
    generate_diagnostics("tests/frame_027048.jpg")