from __future__ import annotations

import numpy as np

from asr.models.base import ASRModel, ModelOutput, ModelState


class ParakeetModel(ASRModel):
    identifier = "parakeet-tdt-0.6b-v3"
    name = "NVIDIA Parakeet-TDT 0.6B v3"
    vendor = "nvidia"
    # Parakeet-TDT 0.6B v3 is multilingual: 25 European languages.
    languages = [
        "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
        "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro",
        "ru", "sk", "sl", "sv", "uk",
    ]
    expected_sr_hz = 16_000
    _hf_id = "nvidia/parakeet-tdt-0.6b-v3"

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def load(self) -> None:
        try:
            import nemo.collections.asr as nemo_asr

            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self._hf_id
            )
            self.state = ModelState.READY
        except Exception as ex:
            self.state = ModelState.FAILED
            self.last_error = f"{type(ex).__name__}: {ex}"
            raise

    def transcribe(self, samples: np.ndarray) -> ModelOutput:
        if self._model is None:
            raise RuntimeError("ParakeetModel.transcribe called before load()")
        result = self._model.transcribe([samples])[0]
        # NeMo returns either a plain str (older API) or a Hypothesis-like
        # object with a .text attribute. An empty .text is the canonical
        # no-speech signal — do NOT fall back to str(result), which would
        # serialize the entire Hypothesis repr into the response.
        if isinstance(result, str):
            text = result
        else:
            text = getattr(result, "text", None) or ""
        return ModelOutput(text=text, language=None)
