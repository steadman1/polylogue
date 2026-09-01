from pathlib import Path
from types import GeneratorType

import pytest

from polylogue.constants import GGUF_MODEL_PATH, MOCK_MESSAGES
from polylogue.helpers.get_supported_model_types import get_supported_model_types
from polylogue.inference.protocols.inference_model import InferenceModel
from polylogue.inference.text_to_text_models.gguf_model import GGUFModel

if not GGUFModel in get_supported_model_types():
    pytest.skip(
        "Skipping GGUFModel tests since lib(s) required to run GGUF models are not available",
        allow_module_level=True,
    )


def test_conforms_protocol_gguf_model() -> None:
    model_path = Path("path/with/no/model")
    model = GGUFModel(model_path)

    assert isinstance(model, InferenceModel)


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
    response = model.generate(MOCK_MESSAGES)

    assert len(response) > 0

    model.destroy()

    assert model.model is None


@pytest.mark.slow
def test_gguf_model_streaming() -> None:
    model_path = GGUF_MODEL_PATH
    model = GGUFModel(model_path)

    model.load()
    stream = model.stream_generate(MOCK_MESSAGES)

    assert isinstance(stream, GeneratorType)
    assert len("".join(stream)) > 0

    model.destroy()

    assert model.model is None
