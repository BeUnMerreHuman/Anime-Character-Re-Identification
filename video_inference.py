from __future__ import annotations

import os
import cv2
import numpy as np
from PIL import Image
from norfair import Detection, Tracker, Video
from norfair.drawing import draw_tracked_boxes

from database import PersonDatabase

DETECT_EVERY     = 4
NORFAIR_DIST_THR = 50
HIT_COUNTER_MAX  = 3

def _mean_emb(embs: list[np.ndarray]) -> np.ndarray:
    arr = np.stack(embs, axis=0)
    m   = arr.mean(axis=0)
    n   = np.linalg.norm(m)
    return m / n if n > 1e-9 else m

def process_video(
    input_path: str,
    detector,
    embedder_processor,
    embedder_session,
    db: PersonDatabase,
    progress_callback=None,
) -> tuple[str, dict]:
    from model import run_detector, run_embedder_batch

    video = Video(input_path=input_path)
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=NORFAIR_DIST_THR,
        hit_counter_max=HIT_COUNTER_MAX,
    )

    uid_info: dict[int, dict] = {}
    uid_embeddings: dict[int, list[np.ndarray]] = {}

    for frame_idx, frame in enumerate(video):
        run_detection = (frame_idx % DETECT_EVERY == 0)
        detections = []

        if run_detection:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes_orig, scores, _ = run_detector(detector, frame_pil)

            crops = []
            if len(boxes_orig) > 0:
                for box, score in zip(boxes_orig, scores):
                    x1, y1, x2, y2 = map(float, box)
                    crops.append(frame_pil.crop([x1, y1, x2, y2]))
                    detections.append(Detection(
                        points=np.array([[x1, y1], [x2, y2]]),
                        data={"score": float(score), "crop": crops[-1]}
                    ))

                embeddings = run_embedder_batch(embedder_processor, embedder_session, crops)
                for det, emb in zip(detections, embeddings):
                    det.data["emb"] = emb

        tracked_objects = tracker.update(detections=detections)

        for obj in tracked_objects:
            if obj.last_detection is not None and "emb" in obj.last_detection.data:
                emb = obj.last_detection.data["emb"]
                crop = obj.last_detection.data["crop"]

                if not hasattr(obj, "db_uid"):
                    db_uid, db_label, _ = db.search(emb)
                    if db_uid is None:
                        db_uid = db.create_identity(emb, crop)
                        db_label = None
                    else:
                        db.add_embedding(db_uid, emb, crop)

                    obj.db_uid = db_uid
                    obj.label = db_label or f"ID:{db_uid}"

                    if db_uid not in uid_info:
                        uid_info[db_uid] = {
                            "label": db_label,
                            "thumbnail": crop,
                            "first_frame": frame_idx,
                        }
                    uid_embeddings.setdefault(db_uid, []).append(emb)
                else:
                    db_uid = obj.db_uid
                    uid_embeddings[db_uid].append(emb)
                    if len(uid_embeddings[db_uid]) % 12 == 0:
                        db.add_embedding(db_uid, emb, crop)

                uid_info[db_uid]["emb_centroid"] = _mean_emb(uid_embeddings[db_uid])

        draw_tracked_boxes(frame, tracked_objects, draw_labels=True)
        video.write(frame)

        if progress_callback and video.total_frames > 0:
            progress_callback(frame_idx / video.total_frames, f"Processing frame {frame_idx}/{video.total_frames}")

    h264_path = video.output_path.replace(".mp4", "_h264.mp4")
    ret = os.system(
        f'ffmpeg -y -i "{video.output_path}" -i "{input_path}" '
        f'-map 0:v:0 -map 1:a:0? -c:v libx264 -crf 23 -preset fast '
        f'-c:a aac -shortest -movflags +faststart "{h264_path}" -loglevel error'
    )
    
    if ret == 0 and os.path.exists(h264_path):
        os.remove(video.output_path)
        final_path = h264_path
    else:
        final_path = video.output_path

    if progress_callback:
        progress_callback(1.0, "Done")

    return final_path, uid_info