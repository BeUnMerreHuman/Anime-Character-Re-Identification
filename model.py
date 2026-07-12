import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps

def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR, fill_color=(128, 128, 128)):

    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), interpolation)
    new_image = Image.new("RGB", (size, size), fill_color)
    pad_w = (size - new_width) // 2
    pad_h = (size - new_height) // 2
    new_image.paste(image, (pad_w, pad_h))
    
    return new_image, ratio, pad_w, pad_h

def load_pipeline(
    detector_onnx="DEIMv2/best_stg2.onnx",
    embedder_onnx="DINOv3/dino_v3.onnx"
):

    available_providers = ort.get_available_providers()

    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("Using GPU (CUDAExecutionProvider)")
    else:
        providers = ["CPUExecutionProvider"]
        print("CUDA not available. Using CPU.")

    detector = ort.InferenceSession(detector_onnx, providers=providers)
    embedder_session = ort.InferenceSession(embedder_onnx, providers=providers)

    embedder_processor = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return detector, embedder_processor, embedder_session

def run_detector(detector_session, original_image, size=640, thrh=0.4):

    resized_im, ratio, pad_w, pad_h = resize_with_aspect_ratio(original_image.convert('RGB'), size)
    orig_size = np.array([[size, size]], dtype=np.int64)
    
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    im_data = transforms(resized_im).unsqueeze(0).numpy()

    output = detector_session.run(
        output_names=None,
        input_feed={'images': im_data, "orig_target_sizes": orig_size}
    )
    
    labels, boxes, scores = output
    
    labels = labels[0]
    boxes = boxes[0]
    scores = scores[0]
    
    mask = scores > thrh
    lab = labels[mask]
    box = boxes[mask]
    scr = scores[mask]
    
    if len(box) == 0:
        return np.empty((0, 4)), np.array([]), np.array([])
        
    boxes_orig = []
    for bb in box:
        orig_x1 = (bb[0] - pad_w) / ratio
        orig_y1 = (bb[1] - pad_h) / ratio
        orig_x2 = (bb[2] - pad_w) / ratio
        orig_y2 = (bb[3] - pad_h) / ratio
        boxes_orig.append([orig_x1, orig_y1, orig_x2, orig_y2])
        
    return np.array(boxes_orig), np.array(scr), np.array(lab)

def run_embedder(embedder_processor, embedder_session, raw_crop, target_size=(448, 448)):
    
    img_padded = ImageOps.pad(raw_crop.convert('RGB'), target_size, color=(128, 128, 128))
    tensor = embedder_processor(img_padded).unsqueeze(0).numpy()
    input_name = embedder_session.get_inputs()[0].name
    outputs = embedder_session.run(None, {input_name: tensor})
    embedding = outputs[0][0] 
    
    return embedding


def run_embedder_batch(embedder_processor, embedder_session, raw_crops, target_size=(448, 448)):
    
    if not raw_crops:
        return []

    tensors = []
    for crop in raw_crops:
        img_padded = ImageOps.pad(crop.convert("RGB"), target_size, color=(128, 128, 128))
        tensors.append(embedder_processor(img_padded))  # shape: (3, H, W)

    # Stack into a single (N, 3, H, W) batch and convert to numpy once
    batch = torch.stack(tensors, dim=0).numpy()

    input_name = embedder_session.get_inputs()[0].name
    outputs = embedder_session.run(None, {input_name: batch})

    # outputs[0] shape: (N, embedding_dim) — split back into a plain list
    return [outputs[0][i] for i in range(len(raw_crops))]