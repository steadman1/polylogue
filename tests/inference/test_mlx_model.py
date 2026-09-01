from pathlib import Path
from types import GeneratorType

import pytest

from polylogue.constants import MLX_MODEL_PATH, MOCK_MESSAGES
from polylogue.helpers.assert_implements_protocol import assert_implements_protocol
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
    assert_implements_protocol(MLXModel, InferenceModel)


def test_invalid_mlx_model() -> None:
    model_path = Path("path/with/no/model")
    model = MLXModel(model_path)

    with pytest.raises(ValueError):
        model.load()


@pytest.mark.slow
def test_mlx_model() -> None:
    model_path = MLX_MODEL_PATH
    model = MLXModel(model_path)

    model.load()
    response = model.generate(MOCK_MESSAGES)

    assert len(response) > 0

    model.destroy()

    with pytest.raises(AttributeError):
        _ = model.model
        _ = model.tokenizer


@pytest.mark.slow
def test_mlx_model_streaming() -> None:
    model_path = MLX_MODEL_PATH
    model = MLXModel(model_path)

    model.load()
    stream = model.stream_generate(MOCK_MESSAGES)

    assert isinstance(stream, GeneratorType)
    assert len("".join(stream)) > 0

    model.destroy()

    with pytest.raises(AttributeError):
        _ = model.model
        _ = model.tokenizer
