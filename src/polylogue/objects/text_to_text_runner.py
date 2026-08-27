from pathlib import Path
from openai.types import Completion, CompletionChoice
from collections.abc import AsyncIterator
from datetime import datetime
import asyncio

class TextToTextRunner:
    model = None
    tokenizer = None

    def __init__(self, model_path: Path) -> None:
        self.engine: ModelEngine = TextToTextEngine(model_path)
        self.model_dir = model_path
        self.model_name = model_path.parts[-1]
        self.model, self.tokenizer = self._load(model_path)

    def _load(self, model_path: Path):
        return self.engine.load(str(model_path))

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


    async def stream_response(self) -> AsyncIterator[Completion]:
        for _ in range(5):
            await asyncio.sleep(1)

            timestamp_seconds = int(datetime.now().timestamp())

            yield Completion(
                id="",
                choices=[],
                created=timestamp_seconds,
                model=self.model_name,
                object="text_completion"
            )
