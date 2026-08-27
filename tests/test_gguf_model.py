from polylogue.objects.text_to_text_models.gguf_model import GGUFModel
from pathlib import Path

def test_gguf_model() -> None:
    model_path = Path("/Users/spencersteadman/Models/ornith-1.5-9b/Ornith-1.5-9B-Q4_K_M.gguf")
    model = GGUFModel(model_path)

    model.load()
    response = model.generate("Q: hello, how are you? A: i am doing ")

    assert len(response) > 0

    model.destroy()

def test_gguf_model_streaming() -> None:
    model_path = Path("/Users/spencersteadman/Models/ornith-1.5-9b/Ornith-1.5-9B-Q4_K_M.gguf")
    model = GGUFModel(model_path)

    model.load()
    response = model.stream_generate("Q: hello, how are you? A: i am doing ")

    assert len(response) > 0

    model.destroy()
