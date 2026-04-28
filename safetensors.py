import torch
from safetensors.torch import save_file

def convert_to_safetensors(pth_path: str, safetensors_path: str):
    print(f"Loading checkpoint from: {pth_path}")
    
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
    
    if "ema" in checkpoint and checkpoint["ema"] is not None:
        print("Extracting EMA weights (optimal for inference)...")
        state_dict = checkpoint["ema"].get("module", checkpoint["ema"])
    elif "model" in checkpoint:
        print("EMA not found. Extracting standard base model weights...")
        state_dict = checkpoint["model"]
    else:
        print("No 'model' or 'ema' key found. Assuming file is a raw state_dict.")
        state_dict = checkpoint

    clean_state_dict = {}
    seen_pointers = set()
    
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            print(f"Skipping non-tensor key: {key}")
            continue
        
        ptr = value.data_ptr()
        if ptr in seen_pointers:
            print(f"Breaking shared memory link for duplicate tensor: {key}")
            clean_state_dict[key] = value.clone().contiguous()
        else:
            seen_pointers.add(ptr)
            clean_state_dict[key] = value.contiguous()

    print(f"Saving stripped, clean weights to: {safetensors_path}")
    save_file(clean_state_dict, safetensors_path)
    print("Conversion complete.")

if __name__ == "__main__":
    INPUT_PTH = "models/model.pth" 
    OUTPUT_SAFETENSORS = "Anime_Character_Detector\model.safetensors"
    
    convert_to_safetensors(INPUT_PTH, OUTPUT_SAFETENSORS)