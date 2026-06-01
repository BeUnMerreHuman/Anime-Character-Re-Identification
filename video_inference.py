from __future__ import annotations

import os
import cv2
import numpy as np
from PIL import Image
from norfair import Detection, Tracker

from database import PersonDatabase

# Run detector every N frames
DETECT_EVERY = 6

# Norfair tracking settings
NORFAIR_DIST_THR = 0.8
HIT_COUNTER_MAX = 15

# Re-ID Settings
REID_SIMILARITY_THRESHOLD = 0.69  

# ── Colour palette ────────────────────────────────────────────────────────────
_PALETTE = [
    (231,  76,  60),   # red
    ( 46, 204, 113),   # green
    ( 52, 152, 219),   # blue
    (241, 196,  15),   # yellow
    (155,  89, 182),   # purple
    (230, 126,  34),   # orange
    (236,  64, 122),   # pink
    ( 39, 174,  96),   # emerald
    (142,  68, 173),   # amethyst
    ( 22, 160, 133),   # green-sea
    (243, 156,  18),   # sunflower
    (192,  57,  43),   # pomegranate
    (  0, 188, 212),   # cyan
    (103,  58, 183),   # deep-purple
    (255, 112,  67),   # deep-orange
    ( 96, 125, 139),   # blue-grey
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

    # ---------- open input with raw OpenCV ----------
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---------- open output writer ----------
    raw_output_path = _build_output_path(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    # ---------- initialise tracker ----------
    tracker = Tracker(
        distance_function="iou",
        distance_threshold=NORFAIR_DIST_THR,
        hit_counter_max=HIT_COUNTER_MAX,
        initialization_delay=0, 
    )

    # Stores person metadata and embeddings
    uid_info: dict[int, dict] = {}
    uid_embeddings: dict[int, list[np.ndarray]] = {}

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        run_detection = (frame_idx % DETECT_EVERY == 0)
        detections: list[Detection] = []

        # ── 1. Run detector periodically ──
        if run_detection:
            frame_pil = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            boxes_orig, scores, _ = run_detector(detector, frame_pil)
            crops: list[Image.Image] = []

            # Create Norfair detections + crops
            for box, score in zip(boxes_orig, scores):
                x1, y1, x2, y2 = map(float, box)

                crop = frame_pil.crop([x1, y1, x2, y2])
                crops.append(crop)

                detections.append(
                    Detection(
                        points=np.array([[x1, y1], [x2, y2]]),
                        data={
                            "score": float(score),
                            "crop": crop,
                        },
                    )
                )

            # Generate embeddings for detected people
            if crops:
                embeddings = run_embedder_batch(
                    embedder_processor,
                    embedder_session,
                    crops,
                )

                for det, emb in zip(detections, embeddings):
                    det.data["emb"] = emb

        # ── 2. Update Tracker ──
        if run_detection:
            tracked_objects = tracker.update(detections=detections)
        else:
            tracked_objects = tracker.update()

        # ── 3. Process tracked identities & Draw ──
        for obj in tracked_objects:
            
            # -- Database Re-ID & Embedding Logic --
            if obj.last_detection is not None and "emb" in obj.last_detection.data:
                emb = obj.last_detection.data["emb"]
                crop = obj.last_detection.data["crop"]
                is_new_emb = not obj.last_detection.data.get("db_inserted", False)

                # First time seeing this tracked object
                if not hasattr(obj, "db_uid"):
                    db_uid, db_label, _ = db.search(emb)

                    # Create new identity in DB if no match is found
                    if db_uid is None:
                        db_uid = db.create_identity(emb, crop)
                        db_label = None
                    else:
                        db.add_embedding(db_uid, emb, crop)

                    obj.db_uid = db_uid
                    obj.label = db_label or f"ID:{db_uid}"
                    obj.last_detection.data["db_inserted"] = True

                    # Save metadata
                    if db_uid not in uid_info:
                        uid_info[db_uid] = {
                            "label": db_label,
                            "thumbnail": crop,
                            "first_frame": frame_idx,
                        }
                    uid_embeddings.setdefault(db_uid, []).append(emb)

                # Existing tracked identity
                else:
                    db_uid = obj.db_uid

                    if is_new_emb:
                        # --- INTERCEPT ID SWITCH ---
                        centroid = uid_info[db_uid].get("emb_centroid")
                        if centroid is not None:
                            # Because _mean_emb handles L2 normalization, dot product = cosine similarity
                            similarity = float(np.dot(emb, centroid))
                            
                            if similarity < REID_SIMILARITY_THRESHOLD:
                                # Semantic mismatch detected. Overwrite Norfair's assignment.
                                db_uid, db_label, _ = db.search(emb)
                                
                                if db_uid is None:
                                    db_uid = db.create_identity(emb, crop)
                                    db_label = None
                                else:
                                    db.add_embedding(db_uid, emb, crop)
                                
                                # Correct the tracker's object metadata moving forward
                                obj.db_uid = db_uid
                                obj.label = db_label or f"ID:{db_uid}"
                                
                                # Setup dictionary space for the newly injected ID
                                if db_uid not in uid_info:
                                    uid_info[db_uid] = {
                                        "label": db_label,
                                        "thumbnail": crop,
                                        "first_frame": frame_idx,
                                    }

                        # Commit the embedding to the corrected/confirmed ID
                        db.add_embedding(db_uid, emb, crop)
                        obj.last_detection.data["db_inserted"] = True
                        
                        # Only append to RAM when a *new* embedding is generated to save memory
                        uid_embeddings.setdefault(db_uid, []).append(emb)

                # Update centroid embedding for the character
                if db_uid in uid_embeddings and uid_embeddings[db_uid]:
                    uid_info[db_uid]["emb_centroid"] = _mean_emb(uid_embeddings[db_uid])

            # -- Drawing Logic --
            pts = obj.estimate  # shape (2, 2): [[x1,y1],[x2,y2]]
            x1, y1 = int(pts[0][0]), int(pts[0][1])
            x2, y2 = int(pts[1][0]), int(pts[1][1])
            
            # Fallback to the tracker ID if the DB UID hasn't been assigned yet
            uid = getattr(obj, "db_uid", obj.id)
            label = getattr(obj, "label", str(uid))
            
            # Fetch the distinct color for this specific ID
            color = get_colour_for_id(uid)

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw the label background
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1
            )
            
            # Draw the text (white text for contrast against colored background)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )

        # ── 4. Write processed frame ──
        writer.write(frame)

        # Progress callback
        if progress_callback and total_frames > 0:
            progress_callback(
                frame_idx / total_frames,
                f"Processing frame {frame_idx}/{total_frames}",
            )

        frame_idx += 1

    cap.release()
    writer.release()

    # ---------- Convert output video to H264 via ffmpeg ----------
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

    # Replace raw output if conversion succeeded
    if ret == 0 and os.path.exists(h264_path):
        os.remove(raw_output_path)
        final_path = h264_path
    else:
        final_path = raw_output_path

    if progress_callback:
        progress_callback(1.0, "Done")

    return final_path, uid_info