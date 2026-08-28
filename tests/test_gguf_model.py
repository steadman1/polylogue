from pathlib import Path
from types import GeneratorType

from polylogue.constants import TEST_PROMPT
from polylogue.models.text_to_text_models.gguf_model import GGUFModel


def test_gguf_model() -> None:
    model_path = Path(
        "/Users/spencersteadman/Models/ornith-1.5-9b/Ornith-1.5-9B-Q4_K_M.gguf"
    )
    model = GGUFModel(model_path)

    model.load()
    response = model.generate(TEST_PROMPT)

    assert len(response) > 0

    model.destroy()


def test_gguf_model_streaming() -> None:
    model_path = Path(
        "/Users/spencersteadman/Models/ornith-1.5-9b/Ornith-1.5-9B-Q4_K_M.gguf"
    )
    model = GGUFModel(model_path)

    model.load()
    stream = model.stream_generate(TEST_PROMPT)

    assert isinstance(stream, GeneratorType)
    assert len("".join(stream)) > 0

    model.destroy()
