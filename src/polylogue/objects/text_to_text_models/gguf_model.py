import gc
from collections.abc import Generator
from pathlib import Path
from typing import final

from llama_cpp import Llama


@final
class GGUFModel:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

        self.model: Llama | None = None

    def load(self) -> None:
        self.model = Llama(
            model_path=str(self.model_path),
            # n_gpu_layers=-1, # Uncomment to use GPU acceleration
            # seed=1337, # Uncomment to set a specific seed
            # n_ctx=2048, # Uncomment to increase the context window
        )

    def destroy(self) -> None:
        self.model.close()
        self.model = None

        gc.collect()

    def generate(self, prompt: str) -> str:
        if not self.model:
            raise Exception("model is null")

        response = self.model.create_completion(
            prompt=prompt, max_tokens=150, stream=False
        )

        return response["choices"][0]["text"]

    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        if not self.model:
            raise Exception("model is null")

        stream = self.model.create_completion(
            prompt=prompt, max_tokens=150, stream=True
        )
        for chunk in stream:
            yield chunk["choices"][0]["text"]
