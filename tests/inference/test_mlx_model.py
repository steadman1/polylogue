from pathlib import Path
from types import GeneratorType

import pytest

from polylogue.constants import MLX_MODEL_PATH
from polylogue.helpers.get_supported_model_types import get_supported_model_types
from polylogue.inference.protocols.inference_model import InferenceModel
from polylogue.inference.text_to_text_models.mlx_model import MLXModel

if not MLXModel in get_supported_model_types():
    pytest.skip(
        "Skipping MLXModels tests since lib(s) required to run MLX models are not available",
        allow_module_level=True,
    )


def test_conforms_protocol_mlx_model() -> None:
    model_path = Path("path/with/no/model")
    model = MLXModel(model_path)

    assert isinstance(model, InferenceModel)


def test_invalid_mlx_model() -> None:
    model_path = Path("path/with/no/model")
    model = MLXModel(model_path)

    with pytest.raises(ValueError):
        model.load()


@pytest.mark.slow
def test_mlx_model() -> None:
    model_path = MLX_MODEL_PATH
    model = MLXModel(model_path)
    messages = [{"role": "user", "content": [{"type": "text", "text": "test"}]}]

    model.load()
    response = model.generate(messages)

    assert len(response) > 0

    model.destroy()

    with pytest.raises(AttributeError):
        _ = model.model
        _ = mode.tokenizer


@pytest.mark.slow
def test_mlx_model_streaming() -> None:
    model_path = MLX_MODEL_PATH
    model = MLXModel(model_path)
    messages = [{"role": "user", "content": [{"type": "text", "text": "test"}]}]

    model.load()
    stream = model.stream_generate(messages)

    assert isinstance(stream, GeneratorType)
    assert len("".join(stream)) > 0

    model.destroy()

    with pytest.raises(AttributeError):
        _ = model.model
        _ = mode.tokenizer
