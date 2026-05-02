import numpy as np
from PIL import ImageDraw, ImageFont
from model import process_image

def _get_font(size: int = 16):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()

def perform_detection(original_image, model, device, db, session_det):
    if original_image is None:
        return None, [], "Upload an image first.", session_det

    session_det.clear()
    boxes, scores, embeddings = process_image(original_image, model, device)

    draw_image = original_image.copy()
    draw = ImageDraw.Draw(draw_image, "RGBA")
    font = _get_font(14)
    detected_options = []

    for idx, (box, score, emb) in enumerate(zip(boxes, scores, embeddings)):
        x1, y1, x2, y2 = [float(v) for v in box]
        thumbnail = original_image.crop((x1, y1, x2, y2))

        uid, assigned_name, similarity = db.search(emb)
        is_known = uid is not None
        display_name = assigned_name if assigned_name else (f"ID: {uid}" if uid else f"ID: New_{idx}")

        temp_id = f"Detection_{idx}"
        session_det[temp_id] = {
            "emb": emb,
            "thumbnail": thumbnail,
            "bbox": [x1, y1, x2, y2],
            "uid": uid,
            "name": assigned_name,
            "similarity": similarity,
            "score": float(score),
            "is_known": is_known,
        }
        detected_options.append((f"{display_name} (Det_{idx})", temp_id))

        box_color = (0, 220, 80) if is_known else (220, 50, 50)
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        draw.text((x1 + 4, max(y1 - 18, 0)), f"{display_name}  {score:.2f}", fill=box_color, font=font)

    status = f"{len(detected_options)} person(s) detected." if detected_options else "No persons detected."
    return draw_image, detected_options, status, session_det

def redraw_image_from_state(original_image, session_det):
    if original_image is None or not session_det:
        return original_image, []

    draw_image = original_image.copy()
    draw = ImageDraw.Draw(draw_image, "RGBA")
    font = _get_font(14)
    detected_options = []

    for choice_key, det in session_det.items():
        x1, y1, x2, y2 = det["bbox"]
        uid = det["uid"]
        assigned_name = det["name"]

        display_name = assigned_name if assigned_name else (f"ID: {uid}" if uid else "ID: New")
        detected_options.append((f"{display_name} ({choice_key})", choice_key))

        box_color = (0, 220, 80) if det["is_known"] else (220, 50, 50)
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        draw.text((x1 + 4, max(y1 - 18, 0)), f"{display_name}  {det['score']:.2f}", fill=box_color, font=font)

    return draw_image, detected_options

def get_detection_metrics(choice_key, session_det):
    if not choice_key:
        return None, None

    det = session_det.get(choice_key)
    if not det:
        return None, None

    score_pct = int(det["score"] * 100)
    sim_val = f"{det['similarity']:.3f}" if det["similarity"] is not None else "N/A"
    
    if det["is_known"]:
        metrics = f"**Matched ID {det['uid']}** — Confidence: {score_pct}% | Cosine Similarity: {sim_val}"
    else:
        metrics = f"**No Match (New Entity)** — Confidence: {score_pct}% | Best Similarity: {sim_val}"

    return metrics, det.get("thumbnail")

def assign_identity(choice_key, session_det, db):
    if not choice_key:
        return "Select a detection first.", session_det, False

    det = session_det.get(choice_key)
    if not det:
        return "Invalid selection.", session_det, False

    uid = db.create_identity(det["emb"], det["thumbnail"])
    det["uid"] = uid
    det["is_known"] = True
    det["name"] = None
    session_det[choice_key] = det

    return f"✓ Assigned new ID {uid} to this detection.", session_det, True

def save_and_update_label(choice_key, session_det, db, new_label):
    if not choice_key:
        return "Select a detection first.", session_det, False

    det = session_det.get(choice_key)
    if not det:
        return "Invalid selection.", session_det, False

    if not det["is_known"]:
        uid = db.create_identity(det["emb"], det["thumbnail"])
        det["uid"] = uid
        det["is_known"] = True
        msg = f"✓ Created new ID {uid}."
    else:
        uid = det["uid"]
        db.add_embedding(uid, det["emb"], det["thumbnail"])
        msg = f"✓ Updated embeddings for ID {uid}."

    if new_label and new_label.strip():
        db.assign_label_and_merge(uid, new_label.strip())
        det["name"] = new_label.strip()
        msg += f" Label set to '{new_label.strip()}'."

    session_det[choice_key] = det

    return msg, session_det, True