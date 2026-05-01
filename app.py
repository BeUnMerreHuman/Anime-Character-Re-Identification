import gradio as gr
from model import load_pipeline
from database import PersonDatabase
import inference

print("Initialising pipeline …")
model, device = load_pipeline()
db = PersonDatabase()
print("Ready.")


def _count_label(n: int) -> str:
    return f"**Total Detections:** {n}"


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
        original_image, model, device, db, session_det
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
        return "", None
    return metrics, thumbnail


def on_assign_new_id(choice_key, session_det, original_image):
    msg, updated_session, success = inference.assign_identity(choice_key, session_det, db)
    
    if success:
        ui_options = gr.update(choices=db.get_ui_options())
        draw_image, updated_detections = inference.redraw_image_from_state(original_image, updated_session)
        dd_update = gr.update(choices=updated_detections, value=choice_key)
    else:
        ui_options = gr.update()
        draw_image = gr.update()
        dd_update = gr.update()

    return msg, ui_options, updated_session, draw_image, dd_update


def on_save_and_label(choice_key, session_det, new_label, original_image):
    msg, updated_session, success = inference.save_and_update_label(choice_key, session_det, db, new_label)
    
    if success:
        ui_options = gr.update(choices=db.get_ui_options())
        draw_image, updated_detections = inference.redraw_image_from_state(original_image, updated_session)
        dd_update = gr.update(choices=updated_detections, value=choice_key)
    else:
        ui_options = gr.update()
        draw_image = gr.update()
        dd_update = gr.update()

    return msg, ui_options, updated_session, draw_image, dd_update


with gr.Blocks(title="Identity Management Dashboard") as demo:

    session_detections = gr.State({})

    with gr.Row():
        gr.Markdown("# Identity Management Dashboard")

    with gr.Row():
        with gr.Column(scale=2, min_width=0):
            with gr.Group():
                gr.Markdown("### Image Input")
                img_in = gr.Image(type="pil", label="Upload Image", show_label=False)
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
                        img_thumb = gr.Image(type="pil", label="Thumbnail", interactive=False, show_label=False, height=100)

                gr.Markdown("---")
                btn_new_id = gr.Button("Assign New ID")
                btn_save = gr.Button("Save Identity", variant="primary")
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

    img_in.change(
        fn=on_image_change,
        inputs=None,
        outputs=[img_out, dd_detections, det_count, lbl_metrics, img_thumb, status_out, session_detections],
    )

    img_in.clear(
        fn=on_image_change,
        inputs=None,
        outputs=[img_out, dd_detections, det_count, lbl_metrics, img_thumb, status_out, session_detections],
    )

    btn_process.click(
        fn=on_analyze_image,
        inputs=[img_in, session_detections],
        outputs=[img_out, dd_detections, det_count, status_out, session_detections],
    )

    dd_detections.change(
        fn=on_select_detection,
        inputs=[dd_detections, session_detections],
        outputs=[lbl_metrics, img_thumb],
    )

    # WIRED CORRECTLY
    btn_new_id.click(
        fn=on_assign_new_id,
        inputs=[dd_detections, session_detections, img_in],
        outputs=[status_out, combo_identity, session_detections, img_out, dd_detections],
    )

    # WIRED CORRECTLY
    btn_save.click(
        fn=on_save_and_label,
        inputs=[dd_detections, session_detections, combo_identity, img_in],
        outputs=[status_out, combo_identity, session_detections, img_out, dd_detections],
    )

if __name__ == "__main__":
    demo.launch()