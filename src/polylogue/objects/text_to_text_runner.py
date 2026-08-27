from pathlib import Path
from mlx_lm import load, generate, stream_generate
from openai.types import Completion, CompletionChoice
from collections.abc import Generator
from datetime import datetime

from time import sleep

class TextToTextRunner:
    model = None
    tokenizer = None

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model_name = model_dir.parts[-1]
        self.model, self.tokenizer = self._load(model_dir)

    def _load(self, model_dir: Path):
        return load(str(model_dir))

    def destroy(self) -> None:
        self.model = None
        self.tokenizer = None

    def await_response(self) -> Completion:
        timestamp_seconds = int(datetime.now().timestamp())

        return Completion(
            id="",
            choices=[],
            created=timestamp_seconds,
            model=self.model_name,
            object="text_completion"
        )


    def stream_response(self) -> Generator[Completion, None, None]:
        for _ in range(5):
            sleep(0.1)

            timestamp_seconds = int(datetime.now().timestamp())

            yield Completion(
                id="",
                choices=[],
                created=timestamp_seconds,
                model=self.model_name,
                object="text_completion"
            )
