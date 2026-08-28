from pathlib import Path
from types import GeneratorType

from polylogue.objects.text_to_text_models.mlx_model import MLXModel


def test_mlx_model() -> None:
    model_path = Path("/Users/spencersteadman/Models/lil-bard/")
    model = MLXModel(model_path)

    model.load()
    response = model.generate("Q: hello, how are you? A: i am doing ")

    assert len(response) > 0

    model.destroy()


def test_mlx_model_streaming() -> None:
    model_path = Path("/Users/spencersteadman/Models/lil-bard/")
    model = MLXModel(model_path)

    model.load()
    stream = model.stream_generate("Q: hello, how are you? A: i am doing ")

    assert isinstance(stream, GeneratorType)
    assert len("".join(stream)) > 0

    model.destroy()
