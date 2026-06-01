from __future__ import annotations

import os
import cv2
import numpy as np
from PIL import Image
from norfair import Detection, Tracker

from database import PersonDatabase

DETECT_EVERY = 6
NORFAIR_DIST_THR = 0.8
HIT_COUNTER_MAX = 15
REID_SIMILARITY_THRESHOLD = 0.69  

_PALETTE = [
    (231,  76,  60), ( 46, 204, 113), ( 52, 152, 219), (241, 196,  15),
    (155,  89, 182), (230, 126,  34), (236,  64, 122), ( 39, 174,  96),
    (142,  68, 173), ( 22, 160, 133), (243, 156,  18), (192,  57,  43),
    (  0, 188, 212), (103,  58, 183), (255, 112,  67), ( 96, 125, 139),
]

def get_colour_for_id(uid: int) -> tuple[int, int, int]:
    return _PALETTE[(uid - 1) % len(_PALETTE)]

def _mean_emb(embs: list[np.ndarray]) -> np.ndarray:
    arr = np.stack(embs, axis=0)
    mean_emb = arr.mean(axis=0)
    norm = np.linalg.norm(mean_emb)
    return mean_emb / norm if norm > 1e-9 else mean_emb

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
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_output_path = _build_output_path(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    tracker = Tracker(
        distance_function="iou",
        distance_threshold=NORFAIR_DIST_THR,
        hit_counter_max=HIT_COUNTER_MAX,
        initialization_delay=0, 
    )

    uid_info: dict[int, dict] = {}
    uid_embeddings: dict[int, list[np.ndarray]] = {}

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        run_detection = (frame_idx % DETECT_EVERY == 0)
        detections: list[Detection] = []

        if run_detection:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes_orig, scores, _ = run_detector(detector, frame_pil)
            crops: list[Image.Image] = []

            for box, score in zip(boxes_orig, scores):
                x1, y1, x2, y2 = map(float, box)
                crop = frame_pil.crop([x1, y1, x2, y2])
                crops.append(crop)

                detections.append(
                    Detection(
                        points=np.array([[x1, y1], [x2, y2]]),
                        data={"score": float(score), "crop": crop},
                    )
                )

            if crops:
                embeddings = run_embedder_batch(embedder_processor, embedder_session, crops)
                for det, emb in zip(detections, embeddings):
                    det.data["emb"] = emb

        if run_detection:
            tracked_objects = tracker.update(detections=detections)
        else:
            tracked_objects = tracker.update()

        # --- PASS 1: Reserve IDs ---
        active_uids_in_frame = set()
        objects_to_process = []

        for obj in tracked_objects:
            if obj.last_detection is not None and "emb" in obj.last_detection.data:
                emb = obj.last_detection.data["emb"]

                if hasattr(obj, "db_uid"):
                    db_uid = obj.db_uid
                    centroid = uid_info[db_uid].get("emb_centroid")

                    if centroid is not None:
                        similarity = float(np.dot(emb, centroid))
                        if similarity >= REID_SIMILARITY_THRESHOLD:
                            active_uids_in_frame.add(db_uid)
                            objects_to_process.append((obj, emb, False))
                            continue

                objects_to_process.append((obj, emb, True))
            else:
                objects_to_process.append((obj, None, False))

        # --- PASS 2: Assign & Draw ---
        for obj, emb, needs_reid in objects_to_process:
            if emb is not None:
                crop = obj.last_detection.data["crop"]
                is_new_emb = not obj.last_detection.data.get("db_inserted", False)

                if needs_reid:
                    db_uid, db_label, _ = db.search(emb, exclude_uids=active_uids_in_frame)

                    if db_uid is None:
                        db_uid = db.create_identity(emb, crop)
                        db_label = None
                    else:
                        db.add_embedding(db_uid, emb, crop)

                    obj.db_uid = db_uid
                    obj.label = db_label or f"ID:{db_uid}"
                    obj.last_detection.data["db_inserted"] = True

                    if db_uid not in uid_info:
                        uid_info[db_uid] = {
                            "label": db_label,
                            "thumbnail": crop,
                            "first_frame": frame_idx,
                        }
                    uid_embeddings.setdefault(db_uid, []).append(emb)
                    active_uids_in_frame.add(db_uid)

                else:
                    db_uid = obj.db_uid
                    if is_new_emb:
                        db.add_embedding(db_uid, emb, crop)
                        obj.last_detection.data["db_inserted"] = True
                        uid_embeddings.setdefault(db_uid, []).append(emb)

                if db_uid in uid_embeddings and uid_embeddings[db_uid]:
                    uid_info[db_uid]["emb_centroid"] = _mean_emb(uid_embeddings[db_uid])

            pts = obj.estimate
            x1, y1 = int(pts[0][0]), int(pts[0][1])
            x2, y2 = int(pts[1][0]), int(pts[1][1])
            
            uid = getattr(obj, "db_uid", obj.id)
            label = getattr(obj, "label", str(uid))
            color = get_colour_for_id(uid)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
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