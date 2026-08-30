from pathlib import Path
from types import GeneratorType

import pytest

from polylogue.constants import GGUF_MODEL_PATH, TEST_PROMPT
from polylogue.inference.text_to_text_models.gguf_model import GGUFModel


def test_invalid_gguf_model() -> None:
    model_path = Path("path/with/no/model")
    model = GGUFModel(model_path)

    with pytest.raises(ValueError):
        model.load()


@pytest.mark.slow
def test_gguf_model() -> None:
    model_path = GGUF_MODEL_PATH
    model = GGUFModel(model_path)

    model.load()
    response = model.generate(TEST_PROMPT)

    assert len(response) > 0

    model.destroy()

    assert model.model is None


@pytest.mark.slow
def test_gguf_model_streaming() -> None:
    model_path = GGUF_MODEL_PATH
    model = GGUFModel(model_path)

    model.load()
    stream = model.stream_generate(TEST_PROMPT)

    assert isinstance(stream, GeneratorType)
    assert len("".join(stream)) > 0

    model.destroy()

    assert model.model is None
