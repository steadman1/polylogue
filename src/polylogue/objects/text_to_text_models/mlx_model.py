from mlx_lm import load, generate, stream_generate
from pathlib import Path
from collections.abc import AsyncIterator

class MLXModel:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        self.model, self.tokenizer = load(self.model_path)

    def destroy(self) -> None:
        self.model, self.tokenizer = None, None

    def generate(self, prompt: str) -> str:
        return generate(self.model, self.tokenizer, prompt=prompt, verbose=True)

    def stream_generate(self, prompt: str) -> AsyncIterator[str]:

        stream = stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=512
        )
        for chunk in stream:
            yield chunk.text
