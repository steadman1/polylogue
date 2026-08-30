from pathlib import Path
from types import GeneratorType

import pytest

from polylogue.constants import MLX_MODEL_PATH
from polylogue.inference.protocols.inference_model import InferenceModel
from polylogue.inference.text_to_text_models.mlx_model import MLXModel

from polylogue.helpers.can_run_mlx_lm import can_run_mlx_lm

if not can_run_mlx_lm():
    pytest.skip(
        "Skipping MLX tests: Apple Silicon macOS required.",
        allow_module_level=True,
    )

def test_mlx_model_conforms() -> None:
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

    model.load()
    response = model.generate("Q: hello, how are you? A: i am doing ")

    assert len(response) > 0

    model.destroy()

    assert model.model is None
    assert model.tokenizer is None


@pytest.mark.slow
def test_mlx_model_streaming() -> None:
    model_path = MLX_MODEL_PATH
    model = MLXModel(model_path)

    model.load()
    stream = model.stream_generate("Q: hello, how are you? A: i am doing ")

    assert isinstance(stream, GeneratorType)
    assert len("".join(stream)) > 0

    model.destroy()

    assert model.model is None
    assert model.tokenizer is None
