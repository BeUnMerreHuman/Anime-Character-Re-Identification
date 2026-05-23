import numpy as np
from PIL import Image, ImageDraw, ImageFont
from model import run_detector, run_embedder, run_embedder_batch
from database import THRESHOLD_CONFIDENT, THRESHOLD_UNCERTAIN, MIN_EMBEDDINGS_UNCERTAIN


def _get_font(size: int = 16) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except Exception:
        return ImageFont.load_default()


def perform_detection(
    original_image: Image.Image,
    detector,
    embedder_processor,
    embedder_session,
    db,
    session_det: dict,
) -> tuple[Image.Image, list, str, dict]:
    if original_image is None:
        return None, [], "Upload an image first.", session_det

    session_det.clear()

    orig_w, orig_h = original_image.size

    boxes_orig, scores, cls_ids = run_detector(detector, original_image)

    if len(boxes_orig) == 0:
        return original_image.copy(), [], "No characters detected.", session_det

    boxes_orig[:, [0, 2]] = boxes_orig[:, [0, 2]].clip(0, orig_w)
    boxes_orig[:, [1, 3]] = boxes_orig[:, [1, 3]].clip(0, orig_h)

    raw_crops = [
        original_image.crop([float(v) for v in box]) for box in boxes_orig
    ]
    embeddings = run_embedder_batch(embedder_processor, embedder_session, raw_crops)

    draw_image = original_image.copy()
    draw       = ImageDraw.Draw(draw_image, "RGBA")
    font       = _get_font(14)
    detected_options = []

    for idx, (box_orig, score, raw_crop, emb) in enumerate(
        zip(boxes_orig, scores, raw_crops, embeddings)
    ):
        x1, y1, x2, y2 = [float(v) for v in box_orig]

        uid, assigned_name, similarity = db.search(emb)
        is_known = uid is not None

        # Build display name
        if assigned_name:
            display_name = assigned_name
        elif uid:
            display_name = f"ID: {uid}"
        else:
            display_name = f"ID: New_{idx}"

        temp_id = f"Detection_{idx}"
        session_det[temp_id] = {
            "emb":        emb,
            "thumbnail":  raw_crop, 
            "bbox":       [x1, y1, x2, y2],
            "uid":        uid,
            "name":       assigned_name,
            "similarity": similarity,
            "score":      float(score),
            "is_known":   is_known,
        }
        detected_options.append((f"{display_name} (Det_{idx})", temp_id))

        box_color = (0, 220, 80) if is_known else (220, 50, 50)
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        draw.text(
            (x1 + 4, max(y1 - 18, 0)),
            f"{display_name}  {score:.2f}",
            fill=box_color,
            font=font,
        )

    status = (
        f"{len(detected_options)} character(s) detected."
        if detected_options
        else "No characters detected."
    )
    return draw_image, detected_options, status, session_det


def redraw_image_from_state(
    original_image: Image.Image,
    session_det: dict,
) -> tuple[Image.Image, list]:
    if original_image is None or not session_det:
        return original_image, []

    draw_image = original_image.copy()
    draw       = ImageDraw.Draw(draw_image, "RGBA")
    font       = _get_font(14)
    detected_options = []

    for choice_key, det in session_det.items():
        x1, y1, x2, y2 = det["bbox"]
        uid          = det["uid"]
        assigned_name = det["name"]

        if assigned_name:
            display_name = assigned_name
        elif uid:
            display_name = f"ID: {uid}"
        else:
            display_name = "ID: New"

        detected_options.append((f"{display_name} ({choice_key})", choice_key))

        box_color = (0, 220, 80) if det["is_known"] else (220, 50, 50)
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        draw.text(
            (x1 + 4, max(y1 - 18, 0)),
            f"{display_name}  {det['score']:.2f}",
            fill=box_color,
            font=font,
        )

    return draw_image, detected_options


def get_detection_metrics(
    choice_key: str,
    session_det: dict,
) -> tuple[str | None, Image.Image | None]:
    if not choice_key:
        return None, None

    det = session_det.get(choice_key)
    if not det:
        return None, None

    score_pct = int(det["score"] * 100)
    sim_val   = f"{det['similarity']:.3f}" if det["similarity"] is not None else "N/A"

    if det["is_known"]:
        zone = (
            "confident"
            if det["similarity"] is not None and det["similarity"] >= THRESHOLD_CONFIDENT
            else "uncertain (voted)"
        )
        metrics = (
            f"**Matched ID {det['uid']}** *(zone: {zone})* — "
            f"Confidence: {score_pct}% | Similarity: {sim_val} | "
            f"Threshold: ≥{THRESHOLD_CONFIDENT} confident / "
            f"≥{THRESHOLD_UNCERTAIN}+{MIN_EMBEDDINGS_UNCERTAIN} embs uncertain"
        )
    else:
        metrics = (
            f"**No Match (New Character)** — "
            f"Confidence: {score_pct}% | Best Similarity: {sim_val}"
        )

    return metrics, det.get("thumbnail")


def assign_identity(
    choice_key: str,
    session_det: dict,
    db,
) -> tuple[str, dict, bool]:
    if not choice_key:
        return "Select a detection first.", session_det, False

    det = session_det.get(choice_key)
    if not det:
        return "Invalid selection.", session_det, False

    uid = db.create_identity(det["emb"], det["thumbnail"])
    det.update({"uid": uid, "is_known": True, "name": None})
    session_det[choice_key] = det

    return f"✓ Assigned new ID {uid} to this detection.", session_det, True


def save_and_update_label(
    choice_key: str,
    session_det: dict,
    db,
    new_label: str,
) -> tuple[str, dict, bool]:
    if not choice_key:
        return "Select a detection first.", session_det, False

    det = session_det.get(choice_key)
    if not det:
        return "Invalid selection.", session_det, False

    if not det["is_known"]:
        uid = db.create_identity(det["emb"], det["thumbnail"])
        det.update({"uid": uid, "is_known": True})
        msg = f"✓ Created new ID {uid}."
    else:
        uid = det["uid"]
        db.add_embedding(uid, det["emb"], det["thumbnail"])
        emb_count = db.get_embedding_count(uid)
        msg = f"✓ Updated embeddings for ID {uid} ({emb_count} total)."

    if new_label and new_label.strip():
        db.assign_label_and_merge(uid, new_label.strip())
        det["name"] = new_label.strip()
        msg += f" Label set to '{new_label.strip()}'."

    session_det[choice_key] = det
    return msg, session_det, True