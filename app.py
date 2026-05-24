"""
app.py
──────
Identity Management Dashboard
Supports both image and video inputs.
"""

import threading
import gradio as gr

from model import load_pipeline
from database import PersonDatabase
import inference
from video_inference import process_video, colour_for_id

print("Initialising pipeline …")
detector, embedder_processor, embedder_session = load_pipeline()
db = PersonDatabase()
print("Ready.")


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _count_label(n: int) -> str:
    return f"**Total Detections:** {n}"


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE tab callbacks  (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

def on_image_change():
    return (
        None,
        gr.update(choices=[], value=None),
        _count_label(0),
        "Select a detection to see metrics.",
        None,
        "",
        {},
    )


def on_analyze_image(original_image, session_det):
    draw_image, detected_options, status, updated_session = inference.perform_detection(
        original_image,
        detector,
        embedder_processor,
        embedder_session,
        db,
        session_det,
    )

    if draw_image is None:
        return (
            None,
            gr.update(choices=[], value=None),
            _count_label(0),
            status,
            updated_session,
        )

    initial_choice = detected_options[0][1] if detected_options else None
    return (
        draw_image,
        gr.update(choices=detected_options, value=initial_choice),
        _count_label(len(detected_options)),
        status,
        updated_session,
    )


def on_select_detection(choice_key, session_det):
    metrics, thumbnail = inference.get_detection_metrics(choice_key, session_det)
    if metrics is None:
        return "", None, gr.update()
    det   = session_det.get(choice_key, {})
    label = det.get("label") or det.get("name")
    identity_update = gr.update(value=label) if label else gr.update()
    return metrics, thumbnail, identity_update


def on_assign_new_id(choice_key, session_det, original_image):
    msg, updated_session, success = inference.assign_identity(choice_key, session_det, db)

    if success:
        ui_options  = gr.update(choices=db.get_ui_options())
        draw_image, updated_detections = inference.redraw_image_from_state(original_image, updated_session)
        dd_update   = gr.update(choices=updated_detections, value=choice_key)
    else:
        ui_options  = gr.update()
        draw_image  = gr.update()
        dd_update   = gr.update()

    return msg, ui_options, updated_session, draw_image, dd_update


def on_save_and_label(choice_key, session_det, new_label, original_image):
    msg, updated_session, success = inference.save_and_update_label(choice_key, session_det, db, new_label)

    if success:
        ui_options  = gr.update(choices=db.get_ui_options())
        draw_image, updated_detections = inference.redraw_image_from_state(original_image, updated_session)
        dd_update   = gr.update(choices=updated_detections, value=choice_key)
    else:
        ui_options  = gr.update()
        draw_image  = gr.update()
        dd_update   = gr.update()

    return msg, ui_options, updated_session, draw_image, dd_update


# ──────────────────────────────────────────────────────────────────────────────
# VIDEO tab callbacks
# ──────────────────────────────────────────────────────────────────────────────

def on_video_change():
    """Reset all video-tab state when a new file is loaded."""
    return (
        None,             # processed_video
        {},               # vid_uid_info  state
        gr.update(choices=[], value=None),  # vid_dd
        "Upload a video and click Process.",  # vid_status
        None,             # vid_thumb
        "",               # vid_identity
        0,                # vid_first_frame state
    )


def on_process_video(video_path, progress=gr.Progress(track_tqdm=False)):
    """Run the full video pipeline and return annotated video + id info."""
    if video_path is None:
        return (
            None,
            {},
            gr.update(choices=[], value=None),
            "Upload a video first.",
            None,
            "",
            0,
        )

    progress(0, desc="Starting …")

    uid_info_result = {}

    def _cb(frac, msg):
        progress(frac, desc=msg)

    try:
        out_path, uid_info_result = process_video(
            video_path,
            detector,
            embedder_processor,
            embedder_session,
            db,
            progress_callback=_cb,
        )
    except Exception as e:
        return (
            None,
            {},
            gr.update(choices=[], value=None),
            f"Error: {e}",
            None,
            "",
            0,
        )

    # Build dropdown options  (one entry per unique db uid)
    dd_choices = []
    for uid, info in sorted(uid_info_result.items()):
        label   = info.get("label") or f"ID {uid}"
        hex_col = _rgb_to_hex(colour_for_id(uid))
        dd_choices.append((f"● {label}  [ID {uid}]", str(uid)))

    initial_uid = dd_choices[0][1] if dd_choices else None
    status = f"✓ Done — {len(uid_info_result)} unique identities detected."

    return (
        out_path,
        uid_info_result,
        gr.update(choices=dd_choices, value=initial_uid),
        status,
        None,
        "",
        0,
    )


def on_vid_select_id(uid_str, uid_info):
    """Show thumbnail + first-frame info when an ID is selected."""
    if not uid_str or not uid_info:
        return None, "", 0

    uid  = int(uid_str)
    info = uid_info.get(uid)
    if info is None:
        return None, "", 0

    label = info.get("label") or f"ID {uid}"
    first = info.get("first_frame", 0)

    # Convert centroid thumbnail to PIL for display
    thumb = info.get("thumbnail")
    if thumb is not None and not hasattr(thumb, "mode"):
        # it's a numpy array
        from PIL import Image
        thumb = Image.fromarray(thumb)

    colour_hex = _rgb_to_hex(colour_for_id(uid))
    info_md = (
        f"**{label}**  "
        f"<span style='color:{colour_hex}'>■</span>  "
        f"First appears at frame **{first}**"
    )

    return thumb, info_md, first


def on_vid_save_label(uid_str, new_label, uid_info):
    """Assign a label to a video-detected identity in the DB."""
    if not uid_str:
        return "Select an identity first.", gr.update(), uid_info

    uid = int(uid_str)
    if not new_label or not new_label.strip():
        return "Enter a label first.", gr.update(), uid_info

    label = new_label.strip()
    db.assign_label_and_merge(uid, label)

    if uid in uid_info:
        uid_info[uid]["label"] = label

    ui_opts = gr.update(choices=db.get_ui_options())
    return f"✓ Saved label '{label}' for ID {uid}.", ui_opts, uid_info


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Identity Management Dashboard",
) as demo:

    # ── Shared state ───────────────────────────────────────────────────────────
    session_detections = gr.State({})
    vid_uid_info       = gr.State({})
    vid_first_frame    = gr.State(0)

    gr.Markdown("# Identity Management Dashboard")

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════════
        # IMAGE tab
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🖼  Image"):
            with gr.Row():
                with gr.Column(scale=2, min_width=0):
                    with gr.Group():
                        gr.Markdown("### Image Input")
                        img_in      = gr.Image(type="pil", label="Upload Image", show_label=False)
                        btn_process = gr.Button("Run Detection", variant="primary")

                    with gr.Group():
                        gr.Markdown("### Detection Output")
                        img_out = gr.Image(type="pil", label="Processed Output", interactive=False, show_label=False)

                with gr.Column(scale=1, min_width=360):
                    with gr.Group():
                        gr.Markdown("### Review & Assign")
                        det_count = gr.Markdown(_count_label(0))

                        gr.Markdown("#### Select Detection")
                        dd_detections = gr.Dropdown(label="Detections", choices=[], interactive=True, show_label=False)

                        with gr.Row():
                            with gr.Column(scale=3):
                                lbl_metrics = gr.Markdown("Select a detection to see metrics.")
                            with gr.Column(scale=2):
                                img_thumb = gr.Image(type="pil", label="Thumbnail", interactive=False,
                                                     show_label=False, height=100)

                        gr.Markdown("---")
                        btn_new_id = gr.Button("Assign New ID")
                        btn_save   = gr.Button("Save Identity", variant="primary")
                        gr.Markdown("---")

                        gr.Markdown("#### Add / Update Label")
                        combo_identity = gr.Dropdown(
                            label="Identities",
                            choices=db.get_ui_options(),
                            allow_custom_value=True,
                            interactive=True,
                            show_label=False,
                            filterable=True,
                        )
                        status_out = gr.Markdown("")

            # Image event wiring
            img_in.change(
                fn=on_image_change, inputs=None,
                outputs=[img_out, dd_detections, det_count, lbl_metrics, img_thumb, status_out, session_detections],
            )
            img_in.clear(
                fn=on_image_change, inputs=None,
                outputs=[img_out, dd_detections, det_count, lbl_metrics, img_thumb, status_out, session_detections],
            )
            btn_process.click(
                fn=on_analyze_image, inputs=[img_in, session_detections],
                outputs=[img_out, dd_detections, det_count, status_out, session_detections],
            )
            dd_detections.change(
                fn=on_select_detection, inputs=[dd_detections, session_detections],
                outputs=[lbl_metrics, img_thumb, combo_identity],
            )
            btn_new_id.click(
                fn=on_assign_new_id, inputs=[dd_detections, session_detections, img_in],
                outputs=[status_out, combo_identity, session_detections, img_out, dd_detections],
            )
            btn_save.click(
                fn=on_save_and_label, inputs=[dd_detections, session_detections, combo_identity, img_in],
                outputs=[status_out, combo_identity, session_detections, img_out, dd_detections],
            )

        # ══════════════════════════════════════════════════════════════════════
        # VIDEO tab
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🎬  Video"):
            with gr.Row():
                # Left column – video in / out
                with gr.Column(scale=2, min_width=0):
                    with gr.Group():
                        gr.Markdown("### Video Input")
                        vid_in      = gr.Video(label="Upload Video", show_label=False)
                        btn_vid_proc = gr.Button("Process Video", variant="primary")

                    with gr.Group():
                        gr.Markdown("### Annotated Output")
                        vid_out     = gr.Video(
                            label="Processed Video",
                            show_label=False,
                            interactive=False,
                            autoplay=False,
                        )

                    vid_status  = gr.Markdown("Upload a video and click Process.")

                # Right column – identity review
                with gr.Column(scale=1, min_width=360):
                    with gr.Group():
                        gr.Markdown("### Identities Detected")
                        gr.Markdown(
                            "_Each colour corresponds to one unique ID. "
                            "Select an ID to see its thumbnail and first appearance._"
                        )
                        vid_dd = gr.Dropdown(
                            label="Unique Identities",
                            choices=[],
                            interactive=True,
                            show_label=False,
                        )

                        vid_id_info   = gr.Markdown("Select an identity above.")
                        vid_thumb     = gr.Image(
                            type="pil", label="Centroid Thumbnail",
                            interactive=False, show_label=True, height=150,
                        )

                        gr.Markdown("---")
                        gr.Markdown("#### Assign / Update Label")
                        vid_identity  = gr.Dropdown(
                            label="Label",
                            choices=db.get_ui_options(),
                            allow_custom_value=True,
                            interactive=True,
                            show_label=False,
                            filterable=True,
                        )
                        btn_vid_save  = gr.Button("Save Label", variant="primary")
                        vid_label_status = gr.Markdown("")

            # Video event wiring
            vid_in.change(
                fn=on_video_change,
                inputs=None,
                outputs=[vid_out, vid_uid_info, vid_dd, vid_status, vid_thumb,
                         vid_identity, vid_first_frame],
            )
            vid_in.clear(
                fn=on_video_change,
                inputs=None,
                outputs=[vid_out, vid_uid_info, vid_dd, vid_status, vid_thumb,
                         vid_identity, vid_first_frame],
            )
            btn_vid_proc.click(
                fn=on_process_video,
                inputs=[vid_in],
                outputs=[vid_out, vid_uid_info, vid_dd, vid_status, vid_thumb,
                         vid_identity, vid_first_frame],
            )
            vid_dd.change(
                fn=on_vid_select_id,
                inputs=[vid_dd, vid_uid_info],
                outputs=[vid_thumb, vid_id_info, vid_first_frame],
            )
            btn_vid_save.click(
                fn=on_vid_save_label,
                inputs=[vid_dd, vid_identity, vid_uid_info],
                outputs=[vid_label_status, vid_identity, vid_uid_info],
            )

if __name__ == "__main__":
    demo.launch(css="""
    .id-pill {
        display: inline-block;
        border-radius: 4px;
        padding: 2px 8px;
        font-weight: 700;
        color: #fff;
    }
    """)