import numpy as np
from PIL import Image, ImageDraw, ImageFont
from model import run_detector, run_embedder

def _get_font(size: int = 16) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except Exception:
        return ImageFont.load_default()

def letterbox_crop(img: Image.Image, target_size: int = 512, pad_color: tuple = (128, 128, 128)) -> Image.Image:
    w, h = img.size
    max_dim = max(w, h)
    
    squared_img = Image.new("RGB", (max_dim, max_dim), pad_color)
    
    paste_x = (max_dim - w) // 2
    paste_y = (max_dim - h) // 2
    
    squared_img.paste(img, (paste_x, paste_y))
    
    return squared_img.resize((target_size, target_size), Image.Resampling.LANCZOS)

def perform_detection(
    original_image: Image.Image,
    detector,
    embedder_processor,
    embedder_model,
    embedder_device,
    db,
    session_det: dict,
) -> tuple[Image.Image, list, str, dict]:
    if original_image is None:
        return None, [], "Upload an image first.", session_det

    session_det.clear()

    # Get dimensions directly from the PIL image
    orig_w, orig_h = original_image.size

    # Pass the PIL Image directly to the refactored run_detector
    boxes_orig, scores, cls_ids = run_detector(detector, original_image)

    if len(boxes_orig) == 0:
        return original_image.copy(), [], "No characters detected.", session_det

    boxes_orig[:, [0, 2]] = boxes_orig[:, [0, 2]].clip(0, orig_w)
    boxes_orig[:, [1, 3]] = boxes_orig[:, [1, 3]].clip(0, orig_h)

    draw_image = original_image.copy()
    draw = ImageDraw.Draw(draw_image, "RGBA")
    font = _get_font(14)
    detected_options = []

    for idx, (box_orig, score) in enumerate(zip(boxes_orig, scores)):
        x1, y1, x2, y2 = [float(v) for v in box_orig]

        raw_crop = original_image.crop((x1, y1, x2, y2))
        
        processed_crop = letterbox_crop(raw_crop, target_size=512, pad_color=(128, 128, 128))

        emb = run_embedder(embedder_processor, embedder_model, embedder_device, processed_crop)

        uid, assigned_name, similarity = db.search(emb)
        is_known = uid is not None

        display_name = (
            assigned_name if assigned_name
            else (f"ID: {uid}" if uid else f"ID: New_{idx}")
        )

        temp_id = f"Detection_{idx}"
        session_det[temp_id] = {
            "emb": emb,
            "thumbnail": processed_crop,  
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
    draw = ImageDraw.Draw(draw_image, "RGBA")
    font = _get_font(14)
    detected_options = []

    for choice_key, det in session_det.items():
        x1, y1, x2, y2 = det["bbox"]
        uid = det["uid"]
        assigned_name = det["name"]

        display_name = (
            assigned_name if assigned_name
            else (f"ID: {uid}" if uid else "ID: New")
        )
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
    sim_val = f"{det['similarity']:.3f}" if det["similarity"] is not None else "N/A"

    if det["is_known"]:
        metrics = (
            f"**Matched ID {det['uid']}** — "
            f"Confidence: {score_pct}% | Cosine Similarity: {sim_val}"
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
        msg = f"✓ Updated embeddings for ID {uid}."

    if new_label and new_label.strip():
        db.assign_label_and_merge(uid, new_label.strip())
        det["name"] = new_label.strip()
        msg += f" Label set to '{new_label.strip()}'."

    session_det[choice_key] = det
    return msg, session_det, True