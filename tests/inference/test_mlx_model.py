from pathlib import Path
from types import GeneratorType

import pytest

from polylogue.helpers.get_inference_models_dir import get_inference_models_dir
from polylogue.inference.text_to_text_models.mlx_model import MLXModel


def test_invalid_mlx_model() -> None:
    model_path = Path("path/with/no/model")
    model = MLXModel(model_path)

    with pytest.raises(ValueError):
        model.load()


@pytest.mark.slow
def test_mlx_model() -> None:
    model_path = get_inference_models_dir() / "lil-bard"
    model = MLXModel(model_path)

    model.load()
    response = model.generate("Q: hello, how are you? A: i am doing ")

    assert len(response) > 0

    model.destroy()

    assert model.model is None
    assert model.tokenizer is None


@pytest.mark.slow
def test_mlx_model_streaming() -> None:
    model_path = get_inference_models_dir() / "lil-bard"
    model = MLXModel(model_path)

    model.load()
    stream = model.stream_generate("Q: hello, how are you? A: i am doing ")

    assert isinstance(stream, GeneratorType)
    assert len("".join(stream)) > 0

    model.destroy()

    assert model.model is None
    assert model.tokenizer is None
