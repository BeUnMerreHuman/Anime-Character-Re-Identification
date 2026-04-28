import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from model import load_pipeline, process_image
from database import PersonDatabase

# ---------------------------------------------------------------------------
# Global singletons  (initialised once at startup)
# ---------------------------------------------------------------------------

print("Initialising pipeline …")
model, device = load_pipeline()
db = PersonDatabase()
print("Ready.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_font(size: int = 16):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except Exception:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def clear_on_upload(image):
    """Reset all outputs whenever a new image is uploaded."""
    return (
        None,                                                # img_out
        gr.update(choices=[], value=None),                   # dd_detections
        "Awaiting processing…",                              # status_text
        f"### Total Unique Entities: {len(db.identities)}", # lbl_entities
        "",                                                  # lbl_confidence
        gr.update(value=None),                               # combo_identity
        "",                                                  # save_status
        {},                                                  # session_embeddings
        {},                                                  # session_detections
    )


def analyze_image(original_image, session_emb, session_det):
    if original_image is None:
        return (
            None,
            gr.update(choices=[], value=None),
            "Upload an image first.",
            f"### Total Unique Entities: {len(db.identities)}",
            "",
            gr.update(value=None),
            "",
            session_emb,
            session_det,
        )

    session_emb.clear()
    session_det.clear()

    boxes, scores, embeddings = process_image(original_image, model, device)

    draw_image = original_image.copy()
    draw       = ImageDraw.Draw(draw_image, "RGBA")
    font       = _get_font()

    detected_options = []

    for idx, (box, score, raw_emb) in enumerate(zip(boxes, scores, embeddings)):
        x1, y1, x2, y2 = [float(v) for v in box]
        score_val       = float(score)

        emb     = raw_emb / np.linalg.norm(raw_emb)
        temp_id = f"Detection_{idx}"
        session_emb[temp_id] = emb

        assigned_name, dist = db.search(emb)
        is_known     = assigned_name is not None
        display_name = assigned_name if is_known else "Unknown"

        session_det[temp_id] = {
            "name":     display_name,
            "dist":     dist,
            "score":    score_val,
            "is_known": is_known,
        }

        detected_options.append(f"{display_name} (Det_{idx})")

        box_color_fill    = (0, 200, 80, 40)   if is_known else (255, 0, 0, 40)
        box_color_outline = "lime"              if is_known else "red"
        label_bg          = (0, 160, 60, 200)  if is_known else (200, 0, 0, 200)

        draw.rectangle([x1, y1, x2, y2], fill=box_color_fill, outline=box_color_outline, width=2)
        text = f"{display_name} | {score_val:.2f}"

        try:
            bbox   = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = font.getsize(text)

        draw.rectangle(
            [x1, max(0, y1 - text_h - 6), x1 + text_w + 6, y1],
            fill=label_bg,
        )
        draw.text((x1 + 3, max(0, y1 - text_h - 3)), text, fill="white", font=font)

    status         = f"Detected {len(boxes)} person(s)."
    initial_choice = detected_options[0] if detected_options else None

    return (
        draw_image,
        gr.update(choices=detected_options, value=initial_choice),
        status,
        f"### Total Unique Entities: {len(db.identities)}",
        "",
        gr.update(value=None),
        "",
        session_emb,
        session_det,
    )


def on_detection_select(choice, all_choices, session_det):
    if not choice:
        return "", gr.update(value=None)

    idx = all_choices.index(choice) if choice in all_choices else None
    if idx is None:
        return "", gr.update(value=None)

    temp_id = f"Detection_{idx}"
    det     = session_det.get(temp_id)
    if not det:
        return "", gr.update(value=None)

    score_pct = int(det["score"] * 100)
    dist      = det["dist"]

    if det["is_known"] and dist is not None:
        if dist > db.variance_warning:
            confidence_md = (
                f"⚠️ **Low confidence match** — "
                f"Confidence: {score_pct}% (high variance, verify manually)"
            )
        else:
            confidence_md = f"✅ Confidence: {score_pct}%"
        prefill = det["name"]
    else:
        confidence_md = f"❓ Unknown person — Confidence: {score_pct}%"
        prefill       = None

    return confidence_md, gr.update(value=prefill)


def save_identity(choice, all_choices, identity_value, session_emb, session_det):
    _no_change = (
        "No detection selected.",
        f"### Total Unique Entities: {len(db.identities)}",
        gr.update(choices=db.get_names()),
    )

    if not choice:
        return _no_change

    idx = all_choices.index(choice) if choice in all_choices else None
    if idx is None:
        return "Could not find selection.", _no_change[1], _no_change[2]

    temp_id    = f"Detection_{idx}"
    final_name = (identity_value or "").strip()

    if not final_name:
        return (
            "⚠️ Please type or select an identity name.",
            f"### Total Unique Entities: {len(db.identities)}",
            gr.update(choices=db.get_names()),
        )

    emb = session_emb.get(temp_id)
    if emb is None:
        return (
            "❌ Embedding lost. Re-process the image.",
            f"### Total Unique Entities: {len(db.identities)}",
            gr.update(choices=db.get_names()),
        )

    det           = session_det.get(temp_id)
    original_name = det["name"] if det else None

    if original_name and original_name != "Unknown" and final_name != original_name:
        warn = (
            f"⚠️ Identity corrected: was **{original_name}**, "
            f"saved as **{final_name}**. Centroids updated."
        )
    else:
        warn = (
            f"✅ Saved as **{final_name}**. "
            "Centroids updated. Hit 'Process Image' to refresh."
        )

    db.add_embedding(final_name, emb)

    return warn, f"### Total Unique Entities: {len(db.identities)}", gr.update(choices=db.get_names())


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

css = ".confidence-box { font-size: 1.05em; margin: 4px 0 8px 0; }"

with gr.Blocks(theme=gr.themes.Monochrome(), css=css) as demo:
    gr.Markdown("# Zero-Shot Person Re-ID Engine (Centroid-Optimised)")

    session_embeddings = gr.State({})
    session_detections = gr.State({})

    with gr.Row():
        # ---- Left column: images ----
        with gr.Column(scale=2):
            img_in      = gr.Image(type="pil", label="Upload Image")
            btn_process = gr.Button("Process Image", variant="primary")
            img_out     = gr.Image(type="pil", label="Analysis Output", interactive=False)

        # ---- Right column: controls ----
        with gr.Column(scale=1):
            lbl_entities   = gr.Markdown(f"### Total Unique Entities: {len(db.identities)}")
            gr.Markdown("---")
            status_text    = gr.Markdown("Awaiting image…")
            dd_detections  = gr.Dropdown(
                label="Select Detection to Review", choices=[], interactive=True
            )
            lbl_confidence = gr.Markdown("", elem_classes=["confidence-box"])
            gr.Markdown(
                "### Confirm or Correct Identity\n"
                "Search or type a name to assign this detection:"
            )
            combo_identity = gr.Dropdown(
                label="Identity",
                choices=db.get_names(),
                allow_custom_value=True,
                interactive=True,
                filterable=True,
            )
            btn_save    = gr.Button("Save to Database", variant="primary")
            save_status = gr.Markdown("")

    # ---- Wiring ----

    _clear_outputs = [
        img_out, dd_detections, status_text, lbl_entities,
        lbl_confidence, combo_identity, save_status,
        session_embeddings, session_detections,
    ]

    img_in.change(
        fn=clear_on_upload,
        inputs=[img_in],
        outputs=_clear_outputs,
    )

    btn_process.click(
        fn=analyze_image,
        inputs=[img_in, session_embeddings, session_detections],
        outputs=_clear_outputs,
    )

    dd_detections.change(
        fn=on_detection_select,
        inputs=[dd_detections, dd_detections, session_detections],
        outputs=[lbl_confidence, combo_identity],
    )

    btn_save.click(
        fn=save_identity,
        inputs=[dd_detections, dd_detections, combo_identity, session_embeddings, session_detections],
        outputs=[save_status, lbl_entities, combo_identity],
    )


if __name__ == "__main__":
    demo.launch()