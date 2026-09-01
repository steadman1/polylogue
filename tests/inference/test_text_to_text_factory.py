import sys
from pathlib import Path

import pytest

from polylogue.constants import GGUF_MODEL_PATH, MLX_MODEL_PATH
from polylogue.helpers.get_supported_model_types import ModelTypeUnsupportedError
from polylogue.inference.text_to_text_factory import (
    ModelResolutionError,
    TextToTextFactory,
)
from polylogue.inference.text_to_text_models.gguf_model import GGUFModel
from polylogue.inference.text_to_text_models.mlx_model import MLXModel


def test_invalid_path_TextToTextFactory() -> None:
    path = Path("path/with/no/model")

    with pytest.raises(ModelResolutionError):
        _ = TextToTextFactory.from_path(path)


def test_missing_llama_cpp_TextToTextFactory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", None)

    with pytest.raises(ModelTypeUnsupportedError):
        _ = TextToTextFactory.from_path(GGUF_MODEL_PATH)


def test_missing_mlx_TextToTextFactory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlx", None)

    with pytest.raises(ModelTypeUnsupportedError):
        _ = TextToTextFactory.from_path(MLX_MODEL_PATH)


def test_gguf_TextToTextFactory() -> None:
    # known gguf model path
    model = TextToTextFactory.from_path(GGUF_MODEL_PATH)

    assert isinstance(model, GGUFModel)


def test_mlx_TextToTextFactory() -> None:
    # known mlx model path
    model = TextToTextFactory.from_path(MLX_MODEL_PATH)

    assert isinstance(model, MLXModel)
