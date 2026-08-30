from pathlib import Path

from pytest import raises

from polylogue.constants import GGUF_MODEL_PATH, MLX_MODEL_PATH
from polylogue.inference.text_to_text_factory import (
    ModelResolutionError,
    TextToTextFactory,
)
from polylogue.inference.text_to_text_models.gguf_model import GGUFModel
from polylogue.inference.text_to_text_models.mlx_model import MLXModel


def test_throws_TextToTextFactory() -> None:
    path = Path("path/with/no/model")

    with raises(ModelResolutionError):
        _ = TextToTextFactory.from_path(path)


def test_gguf_TextToTextFactory() -> None:
    # known gguf model path
    model = TextToTextFactory.from_path(GGUF_MODEL_PATH)

    assert isinstance(model, GGUFModel)


def test_mlx_TextToTextFactory() -> None:
    # known mlx model path
    model = TextToTextFactory.from_path(MLX_MODEL_PATH)

    assert isinstance(model, MLXModel)
