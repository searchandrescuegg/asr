import pytest

from asr.api.errors import ErrorCode, TranscriptionError
from asr.models.base import ModelState
from asr.models.registry import ModelRegistry
from tests.conftest import SecondaryStubModel, StubModel


def test_register_and_get_default():
    primary = StubModel()
    secondary = SecondaryStubModel()
    primary.load()
    secondary.load()
    registry = ModelRegistry(default_identifier=primary.identifier)
    registry.register(primary)
    registry.register(secondary)

    assert registry.get_default() is primary
    assert registry.get(secondary.identifier) is secondary


def test_get_unknown_model_raises_model_not_found():
    registry = ModelRegistry(default_identifier="stub-en")
    primary = StubModel()
    primary.load()
    registry.register(primary)
    with pytest.raises(TranscriptionError) as exc:
        registry.get("does-not-exist")
    assert exc.value.code == ErrorCode.MODEL_NOT_FOUND
    assert exc.value.details["available"] == ["stub-en"]


def test_failed_model_excluded_from_list_available_but_in_list_all():
    primary = StubModel()
    primary.load()
    failed = SecondaryStubModel()
    failed.state = ModelState.FAILED
    failed.last_error = "synthetic"
    registry = ModelRegistry(default_identifier=primary.identifier)
    registry.register(primary)
    registry.register(failed)

    available = [m.identifier for m in registry.list_available()]
    all_models = [m.identifier for m in registry.list_all()]
    assert primary.identifier in available
    assert failed.identifier not in available
    assert failed.identifier in all_models


def test_get_failed_model_raises_model_unavailable():
    failed = StubModel()
    failed.state = ModelState.FAILED
    failed.last_error = "boom"
    registry = ModelRegistry(default_identifier=failed.identifier)
    registry.register(failed)
    with pytest.raises(TranscriptionError) as exc:
        registry.get(failed.identifier)
    assert exc.value.code == ErrorCode.MODEL_UNAVAILABLE
    assert exc.value.details["last_error"] == "boom"


def test_default_unavailable_raises_no_default_model():
    failed = StubModel()
    failed.state = ModelState.FAILED
    registry = ModelRegistry(default_identifier=failed.identifier)
    registry.register(failed)
    with pytest.raises(TranscriptionError) as exc:
        registry.get_default()
    assert exc.value.code == ErrorCode.NO_DEFAULT_MODEL


def test_default_not_registered_raises_no_default_model():
    registry = ModelRegistry(default_identifier="stub-en")
    with pytest.raises(TranscriptionError) as exc:
        registry.get_default()
    assert exc.value.code == ErrorCode.NO_DEFAULT_MODEL


def test_resolve_passes_through_default_when_none():
    primary = StubModel()
    primary.load()
    registry = ModelRegistry(default_identifier=primary.identifier)
    registry.register(primary)
    assert registry.resolve(None) is primary
    assert registry.resolve(primary.identifier) is primary
