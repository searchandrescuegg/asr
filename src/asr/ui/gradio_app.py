from __future__ import annotations

import asyncio
from pathlib import Path

import gradio as gr
from fastapi import FastAPI

from asr.api.errors import TranscriptionError
from asr.models.base import ModelState


def _load_audio_bytes(path: str | None) -> bytes:
    if not path:
        raise gr.Error("No audio file provided")
    return Path(path).read_bytes()


def mount_ui(app: FastAPI) -> FastAPI:
    def transcribe_handler(audio_path: str | None, model_choice: str | None):
        try:
            data = _load_audio_bytes(audio_path)
        except Exception as ex:
            return (
                gr.update(value=f"Error: {ex}", visible=True),
                gr.update(value="", visible=False),
                gr.update(visible=False),
            )

        pipeline = app.state.pipeline
        chosen = model_choice if model_choice and model_choice != "(default)" else None
        try:
            result = asyncio.run(pipeline.transcribe(data, chosen))
        except TranscriptionError as ex:
            return (
                gr.update(value=f"[{ex.code}] {ex.message}", visible=True),
                gr.update(value="", visible=False),
                gr.update(visible=False),
            )

        if result.no_speech_detected:
            text_value = "No speech detected"
        else:
            text_value = result.text
        caption_md = (
            f"_via `{result.model}` · {result.audio_duration_s:.1f}s audio · "
            f"{result.inference_duration_s:.2f}s inference_"
        )
        return (
            gr.update(value=text_value, visible=True),
            gr.update(value=caption_md, visible=True),
            gr.update(visible=False),
        )

    def available_choices() -> list[str]:
        registry = getattr(app.state, "registry", None)
        if registry is None:
            return ["(default)"]
        choices = ["(default)"] + [
            m.identifier for m in registry.list_all() if m.state == ModelState.READY
        ]
        return choices

    with gr.Blocks(title="ASR — Multi-Model Transcription") as ui:
        gr.Markdown("# Multi-Model Speech Transcription")
        gr.Markdown(
            "Upload an audio file. Pick a model (or leave default). "
            "Result appears below."
        )
        with gr.Row():
            audio_in = gr.Audio(type="filepath", label="Audio file")
            model_dd = gr.Dropdown(
                choices=available_choices(),
                value="(default)",
                label="Model",
                allow_custom_value=False,
            )
        submit = gr.Button("Transcribe", variant="primary")
        # Transcription text only — no model name suffix in the textbox so
        # the user can copy it cleanly. The model identifier and timing
        # appear below the box as a caption.
        result_box = gr.Textbox(
            label="Transcription",
            lines=8,
            visible=True,
        )
        result_caption = gr.Markdown(value="", visible=False)
        ui.load(fn=lambda: gr.update(choices=available_choices()), outputs=model_dd)

        def _show_busy():
            return (
                gr.update(value="Transcribing…", visible=True),
                gr.update(value="", visible=False),
                gr.update(interactive=False),
            )

        submit.click(
            fn=_show_busy,
            outputs=[result_box, result_caption, submit],
        ).then(
            fn=transcribe_handler,
            inputs=[audio_in, model_dd],
            outputs=[result_box, result_caption, submit],
        ).then(
            fn=lambda: gr.update(interactive=True),
            outputs=submit,
        )

    gr.mount_gradio_app(app, ui, path="/")
    return app
