"""
video_inference.py
──────────────────
Processes a video at 24 fps output.
• Detection + embedding runs every 4th frame (≈ 6 fps effective).
• Norfair tracks bounding boxes on all frames.
• Embedding DB re-ID is applied at detection frames to maintain long-term identity.
• Each unique track/DB id gets a deterministic, distinct colour.
• Results are burned into every frame and written to an output file.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import norfair
from norfair import Detection, Tracker
from PIL import Image, ImageDraw, ImageFont

from database import PersonDatabase, THRESHOLD_CONFIDENT, THRESHOLD_UNCERTAIN

# ── Constants ────────────────────────────────────────────────────────────────
OUTPUT_FPS        = 24
DETECT_EVERY      = 4          # run detector + embedder every N frames  → 6 fps
NORFAIR_DIST_THR  = 50         # pixels – Euclidean distance for tracker
MAX_DISAPPEARED   = OUTPUT_FPS * 3   # frames before a track is killed
HIT_COUNTER_MAX   = 3

# ── Colour palette ────────────────────────────────────────────────────────────
# 20 visually distinct colours; we cycle if there are more IDs
_PALETTE = [
    (231,  76,  60),   # red
    ( 46, 204, 113),   # green
    ( 52, 152, 219),   # blue
    (241, 196,  15),   # yellow
    (155,  89, 182),   # purple
    ( 26, 188, 156),   # teal
    (230, 126,  34),   # orange
    (236,  64, 122),   # pink
    ( 52,  73,  94),   # dark-slate
    ( 39, 174,  96),   # emerald
    (142,  68, 173),   # amethyst
    ( 22, 160, 133),   # green-sea
    (243, 156,  18),   # sunflower
    (211,  84,   0),   # pumpkin
    ( 41, 128, 185),   # belize
    (192,  57,  43),   # pomegranate
    (  0, 188, 212),   # cyan
    (103,  58, 183),   # deep-purple
    (255, 112,  67),   # deep-orange
    ( 96, 125, 139),   # blue-grey
]

_id_colour_cache: dict[int, tuple[int, int, int]] = {}

def colour_for_id(uid: int) -> tuple[int, int, int]:
    """Return a stable BGR colour for a given numeric id."""
    if uid not in _id_colour_cache:
        _id_colour_cache[uid] = _PALETTE[(uid - 1) % len(_PALETTE)]
    return _id_colour_cache[uid]


# ── Font helper ───────────────────────────────────────────────────────────────
def _get_cv_font():
    return cv2.FONT_HERSHEY_DUPLEX


# ── Norfair helpers ───────────────────────────────────────────────────────────
def _box_to_norfair(box: list[float]) -> np.ndarray:
    """[x1,y1,x2,y2] → centroid [cx, cy] for Norfair."""
    x1, y1, x2, y2 = box
    return np.array([[( x1 + x2) / 2, (y1 + y2) / 2]], dtype=float)


def euclidean_distance(detection, estimates):
    """Custom distance for Norfair.
    Handles both plain ndarray estimates and TrackedObject estimates
    (norfair ≥ 2.2 passes TrackedObject instances instead of raw arrays).
    """
    if hasattr(estimates, "estimate"):
        # TrackedObject – unwrap to numpy array
        est_points = np.array(estimates.estimate)
    elif hasattr(estimates, "__iter__") and not isinstance(estimates, np.ndarray):
        # list/sequence of TrackedObjects
        est_points = np.array([
            e.estimate if hasattr(e, "estimate") else e for e in estimates
        ])
    else:
        est_points = estimates
    return np.linalg.norm(detection.points - est_points, axis=1)


# ── Embedding centroid ────────────────────────────────────────────────────────
def _mean_emb(embs: list[np.ndarray]) -> np.ndarray:
    arr = np.stack(embs, axis=0)
    m   = arr.mean(axis=0)
    n   = np.linalg.norm(m)
    return m / n if n > 1e-9 else m


# ── Draw a single frame ───────────────────────────────────────────────────────
def _draw_frame(
    frame_bgr: np.ndarray,
    track_info: list[dict],   # list of {box, uid, label, score}
) -> np.ndarray:
    out = frame_bgr.copy()
    font     = _get_cv_font()
    font_scale = 0.55
    thickness  = 2

    for info in track_info:
        x1, y1, x2, y2 = [int(v) for v in info["box"]]
        uid   = info["uid"]
        label = info.get("label") or f"ID:{uid}"
        score = info.get("score", 0.0)
        text  = f"{label}  {score:.2f}"

        bgr = colour_for_id(uid)

        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, thickness)

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), bgr, -1)
        # white text over coloured bg
        cv2.putText(out, text, (x1 + 2, ty - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out


# ── Main processing function ──────────────────────────────────────────────────
def process_video(
    input_path: str,
    detector,
    embedder_processor,
    embedder_session,
    db: PersonDatabase,
    progress_callback=None,   # callable(fraction: float, status: str)
) -> tuple[str, dict]:
    """
    Process *input_path* and return:
      (output_video_path, id_info_dict)

    id_info_dict: {uid: {"label": str|None,
                          "thumbnail": PIL.Image,
                          "first_frame": int,
                          "emb_centroid": np.ndarray}}
    """
    from model import run_detector, run_embedder_batch  # imported here to avoid circular

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or OUTPUT_FPS
    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output file (mp4 with h264)
    out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(out_fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, OUTPUT_FPS, (src_w, src_h))

    # ── Tracker ───────────────────────────────────────────────────────────────
    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=NORFAIR_DIST_THR,
        hit_counter_max=HIT_COUNTER_MAX,
        initialization_delay=0,
    )

    # ── Per-track state ────────────────────────────────────────────────────────
    # track_id (norfair) → db uid
    track_to_uid:   dict[int, int]              = {}
    # db uid → accumulated embeddings (for centroid)
    uid_embeddings: dict[int, list[np.ndarray]] = {}
    # db uid → metadata
    uid_info:       dict[int, dict]             = {}
    # norfair track_id → last known box
    track_box:      dict[int, list[float]]      = {}

    frame_idx = 0

    # We resample the source video to OUTPUT_FPS using frame-dropping / duplication.
    # Simplest strategy: read every frame, write only the frames that map to output timeline.
    output_frame_count = 0
    src_frame_duration = 1.0 / src_fps
    out_frame_duration = 1.0 / OUTPUT_FPS
    next_out_time      = 0.0

    # last detection results (boxes, labels) – reused between detection frames
    last_track_info: list[dict] = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        src_time = frame_idx * src_frame_duration

        # ── Run detection every DETECT_EVERY output frames ────────────────────
        run_detection = (output_frame_count % DETECT_EVERY == 0)

        if run_detection:
            frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            boxes_orig, scores, _ = run_detector(detector, frame_pil)

            norfair_detections = []
            crop_list   = []
            box_list    = []
            score_list  = []

            if len(boxes_orig) > 0:
                # clip boxes
                boxes_orig[:, [0, 2]] = boxes_orig[:, [0, 2]].clip(0, src_w)
                boxes_orig[:, [1, 3]] = boxes_orig[:, [1, 3]].clip(0, src_h)

                for box, score in zip(boxes_orig, scores):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    crop = frame_pil.crop([x1, y1, x2, y2])
                    crop_list.append(crop)
                    box_list.append([x1, y1, x2, y2])
                    score_list.append(float(score))
                    norfair_detections.append(
                        Detection(points=_box_to_norfair([x1, y1, x2, y2]))
                    )

                embeddings = run_embedder_batch(embedder_processor, embedder_session, crop_list)
            else:
                embeddings = []

            # Update tracker
            tracked = tracker.update(detections=norfair_detections)

            # Match norfair tracks to detections by proximity
            track_info_this_frame: list[dict] = []
            used_tracks = set()

            for det_i, (box, score, emb, crop) in enumerate(
                zip(box_list, score_list, embeddings, crop_list)
            ):
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

                # Find closest norfair track not yet used
                best_tid  = None
                best_dist = float("inf")
                for t in tracked:
                    if t.id in used_tracks:
                        continue
                    est = np.array(t.estimate).flatten()
                    tx, ty = float(est[0]), float(est[1])
                    d = math.hypot(cx - tx, cy - ty)
                    if d < best_dist:
                        best_dist = d
                        best_tid  = t.id

                if best_tid is None:
                    continue
                used_tracks.add(best_tid)

                # DB re-ID
                if best_tid not in track_to_uid:
                    db_uid, db_label, sim = db.search(emb)
                    if db_uid is None:
                        db_uid = db.create_identity(emb, crop)
                        db_label = None
                    else:
                        db.add_embedding(db_uid, emb, crop)
                    track_to_uid[best_tid] = db_uid

                    if db_uid not in uid_info:
                        uid_info[db_uid] = {
                            "label":       db_label,
                            "thumbnail":   crop,
                            "first_frame": output_frame_count,
                            "emb_centroid": emb,
                        }
                    uid_embeddings.setdefault(db_uid, []).append(emb)
                else:
                    db_uid = track_to_uid[best_tid]
                    # Accumulate embedding & update centroid
                    uid_embeddings.setdefault(db_uid, []).append(emb)
                    uid_info[db_uid]["emb_centroid"] = _mean_emb(uid_embeddings[db_uid])
                    # Save every 12th embedding to DB (avoid bloat)
                    if len(uid_embeddings[db_uid]) % 12 == 0:
                        db.add_embedding(db_uid, emb, crop)

                    # Keep freshest label from DB
                    if uid_info[db_uid]["label"] is None:
                        rows = db.table.search().where(f"id = {db_uid}").limit(1).to_pandas() if db.table else None
                        if rows is not None and not rows.empty:
                            uid_info[db_uid]["label"] = rows.iloc[0]["label"]

                track_box[best_tid] = box

                track_info_this_frame.append({
                    "box":   box,
                    "uid":   db_uid,
                    "label": uid_info[db_uid]["label"],
                    "score": score,
                })

            # Also carry forward any tracks without a fresh detection
            for t in tracked:
                if t.id in used_tracks:
                    continue
                if t.id in track_to_uid and t.id in track_box:
                    db_uid = track_to_uid[t.id]
                    track_info_this_frame.append({
                        "box":   track_box[t.id],
                        "uid":   db_uid,
                        "label": uid_info.get(db_uid, {}).get("label"),
                        "score": 0.0,
                    })

            last_track_info = track_info_this_frame

        else:
            # Interpolation frame – update norfair without new detections
            tracked = tracker.update(detections=[])
            # Update boxes from tracker estimates
            interp_info = []
            for t in tracked:
                if t.id in track_to_uid:
                    db_uid = track_to_uid[t.id]
                    # Use tracker estimate for box centre, keep last known size
                    if t.id in track_box:
                        old_box = track_box[t.id]
                        bw = old_box[2] - old_box[0]
                        bh = old_box[3] - old_box[1]
                        est = np.array(t.estimate).flatten()
                        tx, ty = float(est[0]), float(est[1])
                        new_box = [tx - bw/2, ty - bh/2, tx + bw/2, ty + bh/2]
                        track_box[t.id] = new_box
                        interp_info.append({
                            "box":   new_box,
                            "uid":   db_uid,
                            "label": uid_info.get(db_uid, {}).get("label"),
                            "score": 0.0,
                        })
            last_track_info = interp_info

        # ── Write output frame(s) that fall within this source frame's window ──
        while next_out_time <= src_time + src_frame_duration / 2:
            annotated = _draw_frame(frame_bgr, last_track_info)
            writer.write(annotated)
            output_frame_count += 1
            next_out_time       = output_frame_count * out_frame_duration

        frame_idx += 1

        if progress_callback and src_total > 0:
            progress_callback(frame_idx / src_total, f"Processing frame {frame_idx}/{src_total}")

    cap.release()
    writer.release()

    # Re-encode with ffmpeg for browser-compatible h264 (if available)
    h264_path = out_path.replace(".mp4", "_h264.mp4")
    ret = os.system(
        f'ffmpeg -y -i "{out_path}" -vcodec libx264 -crf 23 -preset fast '
        f'-movflags +faststart "{h264_path}" -loglevel error'
    )
    if ret == 0 and os.path.exists(h264_path):
        os.remove(out_path)
        final_path = h264_path
    else:
        final_path = out_path

    if progress_callback:
        progress_callback(1.0, "Done")

    return final_path, uid_info