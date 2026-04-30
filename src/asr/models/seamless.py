from __future__ import annotations

import numpy as np

from asr.models.base import ASRModel, ModelOutput, ModelState


class SeamlessModel(ASRModel):
    identifier = "seamless-m4t-v2"
    name = "Facebook Seamless M4T v2 Large"
    vendor = "meta"
    languages = [
        "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "tr",
        "ar", "zh", "ja", "ko", "hi", "vi",
    ]
    expected_sr_hz = 16_000
    _hf_id = "facebook/seamless-m4t-v2-large"

    def __init__(self) -> None:
        super().__init__()
        self._processor = None
        self._model = None
        self._device = "cpu"

    def load(self) -> None:
        try:
            import torch
            from transformers import (
                AutoProcessor,
                SeamlessM4Tv2ForSpeechToText,
            )

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = AutoProcessor.from_pretrained(self._hf_id)
            self._model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
                self._hf_id
            ).to(self._device)
            self._model.train(False)  # inference mode
            self.state = ModelState.READY
        except Exception as ex:
            self.state = ModelState.FAILED
            self.last_error = f"{type(ex).__name__}: {ex}"
            raise

    def transcribe(self, samples: np.ndarray) -> ModelOutput:
        if self._model is None or self._processor is None:
            raise RuntimeError("SeamlessModel.transcribe called before load()")
        import torch

        with torch.no_grad():
            inputs = self._processor(
                audios=samples,
                sampling_rate=self.expected_sr_hz,
                return_tensors="pt",
            ).to(self._device)
            output_tokens = self._model.generate(
                **inputs, tgt_lang="eng"
            )[0]
        text = self._processor.decode(output_tokens, skip_special_tokens=True)
        return ModelOutput(text=text, language="eng")
