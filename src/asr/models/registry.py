from __future__ import annotations

from asr.api.errors import ErrorCode, TranscriptionError
from asr.models.base import ASRModel, ModelState


class ModelRegistry:
    def __init__(self, default_identifier: str) -> None:
        self._models: dict[str, ASRModel] = {}
        self._default_identifier = default_identifier

    @property
    def default_identifier(self) -> str:
        return self._default_identifier

    def register(self, model: ASRModel) -> None:
        self._models[model.identifier] = model

    def list_all(self) -> list[ASRModel]:
        return list(self._models.values())

    def list_available(self) -> list[ASRModel]:
        return [m for m in self._models.values() if m.state == ModelState.READY]

    def get(self, identifier: str) -> ASRModel:
        model = self._models.get(identifier)
        if model is None:
            raise TranscriptionError(
                ErrorCode.MODEL_NOT_FOUND,
                f"model '{identifier}' is not registered",
                details={"available": [m.identifier for m in self.list_available()]},
            )
        if model.state != ModelState.READY:
            raise TranscriptionError(
                ErrorCode.MODEL_UNAVAILABLE,
                f"model '{identifier}' is not available (state={model.state})",
                details={"state": str(model.state), "last_error": model.last_error},
            )
        return model

    def get_default(self) -> ASRModel:
        if self._default_identifier not in self._models:
            raise TranscriptionError(
                ErrorCode.NO_DEFAULT_MODEL,
                f"default model '{self._default_identifier}' is not registered",
            )
        model = self._models[self._default_identifier]
        if model.state != ModelState.READY:
            raise TranscriptionError(
                ErrorCode.NO_DEFAULT_MODEL,
                f"default model '{self._default_identifier}' is not available",
                details={"state": str(model.state), "last_error": model.last_error},
            )
        return model

    def resolve(self, requested: str | None) -> ASRModel:
        if requested is None:
            return self.get_default()
        return self.get(requested)
