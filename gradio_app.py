from __future__ import annotations

from pathlib import Path

import gradio as gr

from demo_runtime import (
    DEFAULT_ARTIFACTS,
    predict_image_and_text,
    predict_text_only,
    process_video_file,
)


ARTIFACTS = DEFAULT_ARTIFACTS


def run_text(text: str):
    result = predict_text_only(text, artifacts=ARTIFACTS)
    return (
        f"label: {result['label']}\n"
        f"confidence: {result['confidence']}\n"
        f"source: {result['source']}"
    )


def run_image(image, text: str, face_weight: float, text_weight: float):
    annotated, summary = predict_image_and_text(
        image,
        text=text,
        artifacts=ARTIFACTS,
        face_weight=face_weight,
        text_weight=text_weight,
    )
    return annotated, summary


def run_video(video, text: str, face_weight: float, text_weight: float, detect_every: float):
    output_path, summary = process_video_file(
        video,
        text=text,
        artifacts=ARTIFACTS,
        face_weight=face_weight,
        text_weight=text_weight,
        detect_every=detect_every,
    )
    return output_path, summary


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Emotion Recognition Demo") as demo:
        gr.Markdown(
            "# Emotion Recognition Demo\n"
            f"Artifacts: `{ARTIFACTS}`\n\n"
            "Tabs below cover text, image/webcam snapshot, and video."
        )

        with gr.Tabs():
            with gr.TabItem("Text"):
                text_in = gr.Textbox(label="Text", lines=3, value="I am feeling happy today")
                text_out = gr.Textbox(label="Prediction", lines=4)
                gr.Button("Predict Text").click(run_text, inputs=text_in, outputs=text_out)

            with gr.TabItem("Image / Webcam"):
                image_in = gr.Image(type="numpy", sources=["upload", "webcam"], label="Image")
                image_text = gr.Textbox(label="Optional text for fusion", lines=2)
                face_weight = gr.Slider(0, 1, value=0.65, step=0.05, label="Face weight")
                text_weight = gr.Slider(0, 1, value=0.35, step=0.05, label="Text weight")
                image_out = gr.Image(type="numpy", label="Annotated output")
                image_summary = gr.Textbox(label="Summary", lines=8)
                gr.Button("Predict Image").click(
                    run_image,
                    inputs=[image_in, image_text, face_weight, text_weight],
                    outputs=[image_out, image_summary],
                )

            with gr.TabItem("Video"):
                video_in = gr.Video(label="Upload video")
                video_text = gr.Textbox(label="Optional text for fusion", lines=2)
                video_face_weight = gr.Slider(0, 1, value=0.65, step=0.05, label="Face weight")
                video_text_weight = gr.Slider(0, 1, value=0.35, step=0.05, label="Text weight")
                detect_every = gr.Slider(0.5, 5.0, value=2.0, step=0.5, label="Detect every (seconds)")
                video_out = gr.Video(label="Annotated video")
                video_summary = gr.Textbox(label="Summary", lines=8)
                gr.Button("Process Video").click(
                    run_video,
                    inputs=[
                        video_in,
                        video_text,
                        video_face_weight,
                        video_text_weight,
                        detect_every,
                    ],
                    outputs=[video_out, video_summary],
                )
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
