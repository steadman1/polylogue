from __future__ import annotations

from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import final

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice

from polylogue.constants import GGUF_TARGET, MLX_TARGET
from polylogue.helpers.generator_check_last import generator_check_last
from polylogue.models.protocols.text_to_text_model import TextToTextModel
from polylogue.models.text_to_text_models.gguf_model import GGUFModel
from polylogue.models.text_to_text_models.mlx_model import MLXModel


@final
class TextToTextEngine:
    def __init__(self, model_path: Path):
        self.model_path: Path = model_path
        self.model_name: str = model_path.parts[-1]
        self.model: TextToTextModel | None = self._choose_model_type()

        if self.model:
            self.model.load()
        # raise exception if model cannot be loaded?
        # else:
        # raise Exception()

    def _choose_model_type(self) -> TextToTextModel | None:
        # need to decide whether to use mlx, llama, ... here based on file type/directory details
        if self.model_path.is_file() and self.model_name.endswith(GGUF_TARGET):
            # *.gguf files are handled by llama cpp
            return GGUFModel(self.model_path)

        if self.model_path.is_dir() and (self.model_path / MLX_TARGET).is_file():
            # mlx should check for a config.json and safetensors in target directory
            return MLXModel(self.model_path)

        return None

    def destroy(self) -> None:
        self.model.destroy()

    # generation should format the raw dictionary into an openai Completion
    def generate(self, prompt: str) -> ChatCompletion:
        timestamp_seconds = int(datetime.now().timestamp())
        response = self.model.generate(prompt)

        choice = Choice(
            finish_reason="stop",
            index=0,
            message=ChatCompletionMessage(content=response, role="assistant"),
        )

        return ChatCompletion(
            id="0",
            choices=[choice],
            created=timestamp_seconds,
            model=self.model_name,
            object="chat.completion",
        )

    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        timestamp_seconds = int(datetime.now().timestamp())
        stream = self.model.stream_generate(prompt)

        for is_last, chunk in generator_check_last(stream):
            choice = ChunkChoice(
                finish_reason="stop" if is_last else None,
                index=0,
                delta=ChoiceDelta(content=chunk, role="assistant"),
            )
            chunk = ChatCompletionChunk(
                id="0",
                choices=[choice],
                created=timestamp_seconds,
                model=self.model_name,
                object="chat.completion.chunk",
            )

            # Serialize Pydantic chunk to JSON string, formatted for SSE
            chunk_json = chunk.model_dump_json()
            yield f"data: {chunk_json}\n\n"
