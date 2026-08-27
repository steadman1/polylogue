from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import final
from datetime import datetime

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta

from polylogue.objects.protocols.text_to_text_model import TextToTextModel

# from polylogue.objects.text_to_text_models.gguf_model import GGUFModel


@final
class TextToTextEngine:

    def __init__(self, model_path: Path):
        self.model_path: Path = model_path
        self.model_name: str = model_path.parts[-1]
        self.model: TextToTextModel | None = self._load()

    def _load(self) -> TextToTextModel | None:
        # need to decide whether to use mlx, llama, ... here based on file type/directory details

        # if self.model_path.is_file() and self.model_name.endswith(GGUF_TARGET):
        #      # *.gguf files are handled by llama cpp
        #     return MLXModel(self.model_path) # GGUFModel()


        # if self.model_path.is_dir() and (self.model_path / MLX_TARGET).is_file():
        #     # mlx should check for a config.json and safetensors in target directory
        #     return MLXModel(self.model_path)

        return None

    def destroy(self) -> None:
        self.engine.destroy()


    # generation should format the raw dictionary into an openai Completion
    def generate(self, prompt: str) -> Completion:
        timestamp_seconds = int(datetime.now().timestamp())
        response = self.engine.generate(prompt)

        choice = Choice(
            finish_reason="stop",
            index=0,
            message=ChatCompletionMessage(
                content=response,
                role="assistant"
            )
        )

        return ChatCompletion(
            id="0",
            choices=[choice],
            created=timestamp_seconds,
            model=self.model_name,
            object="chat.completion"
        )

    def stream_generate(self, prompt: str) -> Generator[ChatCompletionChunk, None, None]:
        timestamp_seconds = int(datetime.now().timestamp())
        stream = self.engine.stream_generate(prompt)

        for chunk in stream:
            choice = ChunkChoice(
                finish_reason="stop",
                index=0,
                delta=ChoiceDelta(
                    content=chunk,
                    role="assistant"
                )
            )

            yield ChatCompletionChunk(
                id="0",
                choices=[choice],
                created=timestamp_seconds,
                model=self.model_name,
                object="chat.completion"
            )
