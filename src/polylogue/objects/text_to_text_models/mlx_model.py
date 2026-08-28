import gc
from collections.abc import Generator
from pathlib import Path

import mlx.core as mlx
from mlx_lm import generate, load, stream_generate


class MLXModel:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        self.model, self.tokenizer = load(self.model_path)

    def destroy(self) -> None:
        del self.model
        del self.tokenizer

        gc.collect()
        mlx.clear_cache()

    def generate(self, prompt: str) -> str:
        return generate(self.model, self.tokenizer, prompt=prompt, verbose=True)

    def stream_generate(self, prompt: str) -> Generator[str, None, None]:

        stream = stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=512
        )
        for chunk in stream:
            yield chunk.text
