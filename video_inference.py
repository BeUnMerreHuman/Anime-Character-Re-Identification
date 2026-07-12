from __future__ import annotations

import os
import cv2
import numpy as np
from PIL import Image

from database import PersonDatabase

REID_SIMILARITY_THRESHOLD = 0.69

_PALETTE = [
    (231,  76,  60), ( 46, 204, 113), ( 52, 152, 219), (241, 196,  15),
    (155,  89, 182), (230, 126,  34), (236,  64, 122), ( 39, 174,  96),
    (142,  68, 173), ( 22, 160, 133), (243, 156,  18), (192,  57,  43),
    (  0, 188, 212), (103,  58, 183), (255, 112,  67), ( 96, 125, 139),
]

def get_colour_for_id(uid: int) -> tuple[int, int, int]:
    return _PALETTE[(uid - 1) % len(_PALETTE)]

def _build_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_output{ext or '.mp4'}"

def process_video(
    input_path: str,
    detector,
    embedder_processor,
    embedder_session,
    db: PersonDatabase,
    progress_callback=None,
) -> tuple[str, dict]:

    from model import run_detector, run_embedder_batch

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_output_path = _build_output_path(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    uid_info: dict[int, dict] = {}
    
    # 1. Initialize In-Memory Matrix Cache
    df_db = db.get_all_records()
    if not df_db.empty:
        cached_embs_matrix = np.stack(df_db["embedding"].values)
        cached_uids = df_db["id"].tolist()
        cached_labels = df_db["label"].tolist()
        next_db_id = int(df_db["id"].max()) + 1
    else:
        cached_embs_matrix = None
        cached_uids = []
        cached_labels = []
        next_db_id = 1

    new_db_records = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # --- DETECTION ---
        boxes_orig, scores, _ = run_detector(detector, frame_pil)
        
        if len(boxes_orig) > 0:
            crops = [frame_pil.crop(tuple(map(float, box))) for box in boxes_orig]
            
            # --- EMBEDDING ---
            embeddings = run_embedder_batch(embedder_processor, embedder_session, crops)
            
            for box, crop, emb in zip(boxes_orig, crops, embeddings):
                x1, y1, x2, y2 = map(int, box)
                
                matched_uid = None
                matched_label = None
                
                # --- MATCHING VIA DOT PRODUCT MATRIX MULTIPLICATION ---
                if cached_embs_matrix is not None:
                    similarities = np.dot(cached_embs_matrix, emb)
                    best_idx = int(np.argmax(similarities))
                    best_sim = float(similarities[best_idx])
                    
                    if best_sim >= REID_SIMILARITY_THRESHOLD:
                        matched_uid = cached_uids[best_idx]
                        matched_label = cached_labels[best_idx]
                
                # Create New ID if unmatched
                if matched_uid is None:
                    matched_uid = next_db_id
                    next_db_id += 1
                    matched_label = None

                # --- UPDATE LOCAL MATRIX CACHE ---
                if cached_embs_matrix is None:
                    cached_embs_matrix = emb.reshape(1, -1)
                else:
                    cached_embs_matrix = np.vstack([cached_embs_matrix, emb])
                
                cached_uids.append(matched_uid)
                cached_labels.append(matched_label)
                
                # --- QUEUE DATA FOR BATCH PUSH ---
                new_db_records.append({
                    "id": matched_uid,
                    "label": matched_label,
                    "thumbnail": crop,
                    "embedding": emb.tolist()
                })

                if matched_uid not in uid_info:
                    uid_info[matched_uid] = {
                        "label": matched_label,
                        "thumbnail": crop,
                        "first_frame": frame_idx
                    }

                # --- DRAWING ---
                label_str = matched_label if matched_label else f"ID:{matched_uid}"
                color = get_colour_for_id(matched_uid)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    frame, label_str, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
                )

        writer.write(frame)

        if progress_callback and total_frames > 0:
            progress_callback(
                frame_idx / total_frames,
                f"Processing frame {frame_idx}/{total_frames}",
            )

        frame_idx += 1

    cap.release()
    writer.release()

    # 2. Batch insert newly found data into Permanent DB
    if new_db_records:
        db.batch_add(new_db_records)

    # Convert Output Stream to H.264 Format
    h264_path = raw_output_path.replace(".mp4", "_h264.mp4")
    ret = os.system(
        f'ffmpeg -y -i "{raw_output_path}" '
        f'-i "{input_path}" '
        f'-map 0:v:0 -map 1:a:0? '
        f'-c:v libx264 -crf 23 -preset fast '
        f'-c:a aac -shortest '
        f'-movflags +faststart '
        f'"{h264_path}" -loglevel error'
    )

    if ret == 0 and os.path.exists(h264_path):
        os.remove(raw_output_path)
        final_path = h264_path
    else:
        final_path = raw_output_path

    if progress_callback:
        progress_callback(1.0, "Done")

    return final_path, uid_info