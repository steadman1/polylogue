from pathlib import Path
from openai.types import Completion, CompletionChoice
from collections.abc import AsyncIterator
from datetime import datetime
import asyncio
from typing import final
from polylogue.objects.model_engine import ModelEngine
from polylogue.objects.text_to_text_engine import TextToTextEngine

@final
class TextToTextRunner:
    engine = None

    def __init__(self, model_path: Path) -> None:
        self.engine: ModelEngine[Completion] = TextToTextEngine(model_path)
        self._load()
        
    def _load(self):
        self.engine.load()

    def destroy(self) -> None:
        self.engine.destroy()

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
