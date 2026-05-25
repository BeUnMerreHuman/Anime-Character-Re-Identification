from __future__ import annotations

import os
import cv2
import numpy as np
from PIL import Image
from norfair import Detection, Tracker, Video
from norfair.drawing import draw_tracked_boxes

from database import PersonDatabase

# Run detector every N frames
DETECT_EVERY = 4

# Norfair tracking settings
NORFAIR_DIST_THR = 50
HIT_COUNTER_MAX = 3


def _mean_emb(embs: list[np.ndarray]) -> np.ndarray:
    """Compute normalized centroid embedding."""
    
    arr = np.stack(embs, axis=0)
    mean_emb = arr.mean(axis=0)

    norm = np.linalg.norm(mean_emb)
    return mean_emb / norm if norm > 1e-9 else mean_emb


def process_video(
    input_path: str,
    detector,
    embedder_processor,
    embedder_session,
    db: PersonDatabase,
    progress_callback=None,
) -> tuple[str, dict]:

    from model import run_detector, run_embedder_batch

    # Load video and initialize tracker
    video = Video(input_path=input_path)

    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=NORFAIR_DIST_THR,
        hit_counter_max=HIT_COUNTER_MAX,
    )

    # Stores person metadata and embeddings
    uid_info: dict[int, dict] = {}
    uid_embeddings: dict[int, list[np.ndarray]] = {}

    # Process frames one by one
    for frame_idx, frame in enumerate(video):

        run_detection = (frame_idx % DETECT_EVERY == 0)
        detections = []

        # Run detector periodically
        if run_detection:

            frame_pil = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            boxes_orig, scores, _ = run_detector(detector, frame_pil)

            crops = []

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

        # Update tracker with detections
        tracked_objects = tracker.update(detections=detections)

        # Process tracked identities
        for obj in tracked_objects:

            if (
                obj.last_detection is None
                or "emb" not in obj.last_detection.data
            ):
                continue

            emb = obj.last_detection.data["emb"]
            crop = obj.last_detection.data["crop"]

            is_new_emb = not obj.last_detection.data.get(
                "db_inserted",
                False,
            )

            # First time seeing this tracked object
            if not hasattr(obj, "db_uid"):

                db_uid, db_label, _ = db.search(emb)

                # Create new identity if not found
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
                uid_embeddings[db_uid].append(emb)

                if is_new_emb:
                    db.add_embedding(db_uid, emb, crop)
                    obj.last_detection.data["db_inserted"] = True

            # Update centroid embedding
            uid_info[db_uid]["emb_centroid"] = _mean_emb(
                uid_embeddings[db_uid]
            )

        # Draw tracked boxes on frame
        draw_tracked_boxes(
            frame,
            tracked_objects,
            draw_labels=True,
        )

        # Write processed frame
        video.write(frame)

        # Progress callback
        if progress_callback and video.total_frames > 0:
            progress_callback(
                frame_idx / video.total_frames,
                f"Processing frame {frame_idx}/{video.total_frames}",
            )

    # Convert output video to H264
    h264_path = video.output_path.replace(".mp4", "_h264.mp4")

    ret = os.system(
        f'ffmpeg -y -i "{video.output_path}" '
        f'-i "{input_path}" '
        f'-map 0:v:0 -map 1:a:0? '
        f'-c:v libx264 -crf 23 -preset fast '
        f'-c:a aac -shortest '
        f'-movflags +faststart '
        f'"{h264_path}" -loglevel error'
    )

    # Replace original output if conversion succeeded
    if ret == 0 and os.path.exists(h264_path):

        os.remove(video.output_path)
        final_path = h264_path

    else:
        final_path = video.output_path

    if progress_callback:
        progress_callback(1.0, "Done")

    return final_path, uid_info