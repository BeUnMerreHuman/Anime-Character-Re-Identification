from __future__ import annotations

import os
import cv2
import numpy as np
from PIL import Image

from database import PersonDatabase

REID_SIMILARITY_THRESHOLD = 0.69
IOU_TRACKING_THRESHOLD    = 0.30

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

def compute_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
        
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return float(inter_area / union_area) if union_area > 0 else 0.0

def greedy_assignment_uids(sim_matrix: np.ndarray, threshold: float, db_uids: list[int]) -> dict[int, int]:
    """Matches crops to DB embeddings enforcing 1-to-1 crop and unique UID constraints."""
    matched_crops = {}
    if sim_matrix.size == 0:
        return matched_crops
        
    sims = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            if sim_matrix[i, j] >= threshold:
                sims.append((sim_matrix[i, j], i, j))
                
    sims.sort(key=lambda x: x[0], reverse=True)
    
    used_crops = set()
    used_uids = set()
    
    for sim, i, j in sims:
        uid = db_uids[j]
        if i not in used_crops and uid not in used_uids:
            matched_crops[i] = j
            used_crops.add(i)
            used_uids.add(uid)
            
    return matched_crops

def greedy_assignment_iou(iou_matrix: np.ndarray, threshold: float) -> dict[int, int]:
    """Matches bounding boxes to previous tracks based on Intersection over Union."""
    matched_boxes = {}
    if iou_matrix.size == 0:
        return matched_boxes
        
    ious = []
    for i in range(iou_matrix.shape[0]):
        for j in range(iou_matrix.shape[1]):
            if iou_matrix[i, j] >= threshold:
                ious.append((iou_matrix[i, j], i, j))
                
    ious.sort(key=lambda x: x[0], reverse=True)
    
    used_new = set()
    used_trk = set()
    
    for iou, i, j in ious:
        if i not in used_new and j not in used_trk:
            matched_boxes[i] = j
            used_new.add(i)
            used_trk.add(j)
            
    return matched_boxes

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
    active_tracks  = []  # Holds dicts: {"box": list, "uid": int, "label": str}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # --- DETECTION (Runs Every Frame) ---
        boxes_orig, scores, _ = run_detector(detector, frame_pil)
        current_tracks = []
        
        is_keyframe = (frame_idx % 6 == 0)
        
        if len(boxes_orig) > 0:
            if is_keyframe:
                crops = [frame_pil.crop(tuple(map(float, box))) for box in boxes_orig]
                
                # --- EMBEDDING (Runs Every 6th Frame) ---
                embeddings = run_embedder_batch(embedder_processor, embedder_session, crops)
                emb_array  = np.stack(embeddings)
                
                # --- MATCHING VIA UNIQUE BATCH ASSIGNMENT ---
                if cached_embs_matrix is not None:
                    sim_matrix = np.dot(emb_array, cached_embs_matrix.T)
                    matched_crops = greedy_assignment_uids(sim_matrix, REID_SIMILARITY_THRESHOLD, cached_uids)
                else:
                    matched_crops = {}
                    
                for i, (box, emb, crop) in enumerate(zip(boxes_orig, embeddings, crops)):
                    if i in matched_crops:
                        db_idx = matched_crops[i]
                        matched_uid = cached_uids[db_idx]
                        matched_label = cached_labels[db_idx]
                    else:
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

                    current_tracks.append({
                        "box": box.tolist(),
                        "uid": matched_uid,
                        "label": matched_label
                    })
            else:
                # --- IOU TRACKING (Intermediate Frames) ---
                if len(active_tracks) > 0:
                    iou_matrix = np.zeros((len(boxes_orig), len(active_tracks)))
                    for i, box in enumerate(boxes_orig):
                        for j, track in enumerate(active_tracks):
                            iou_matrix[i, j] = compute_iou(box, track["box"])
                            
                    matched_boxes = greedy_assignment_iou(iou_matrix, IOU_TRACKING_THRESHOLD)
                else:
                    matched_boxes = {}
                    
                for i, box in enumerate(boxes_orig):
                    if i in matched_boxes:
                        track_idx = matched_boxes[i]
                        track = active_tracks[track_idx]
                        current_tracks.append({
                            "box": box.tolist(),
                            "uid": track["uid"],
                            "label": track["label"]
                        })
                    else:
                        # Unmatched objects in intermediate frames get a temporary state
                        current_tracks.append({
                            "box": box.tolist(),
                            "uid": -1,
                            "label": "Pending"
                        })
                        
            active_tracks = current_tracks
            
            # --- DRAWING ---
            for trk in current_tracks:
                box = trk["box"]
                uid = trk["uid"]
                label = trk["label"]
                x1, y1, x2, y2 = map(int, box)
                
                if uid == -1:
                    color = (128, 128, 128)
                    label_str = "Pending"
                else:
                    color = get_colour_for_id(uid)
                    label_str = label if label else f"ID:{uid}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    frame, label_str, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
                )
        else:
            active_tracks = []

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