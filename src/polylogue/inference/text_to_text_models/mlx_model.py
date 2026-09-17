import gc
import time
import uuid
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any, final

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from polylogue.inference.chat_templates.universal import CHATML_TEMPLATE


# Models should only be created using a factory
@final
class MLXModel:
    def __init__(
        self,
        model_id: str,
        model_path: Path,
        max_tokens: int = 512,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.max_tokens = max_tokens

        self.model: Any = None
        self.tokenizer: Any = None

    def load(self) -> None:
        from mlx_lm import load

        self.model, self.tokenizer = load(str(self.model_path))  # type: ignore
        self.tokenizer.chat_template = CHATML_TEMPLATE

    def destroy(self) -> None:
        import mlx.core as mlx

        self.model = None
        self.tokenizer = None

        _ = gc.collect()
        mlx.clear_cache()

    def generate(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None = None,
    ) -> ChatCompletion:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        from mlx_lm import generate

        prompt = self.tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools) if tools else None,
            tokenize=False,
            add_generation_prompt=True,
        )

        response: str = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            verbose=False,
        )

        (id, time) = self._get_id_and_created()
        return self._create_chat_completion(response, id, time)

    def stream_generate(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None = None,
    ) -> Generator[ChatCompletionChunk, None, None]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        import mlx.core as mlx_core
        from mlx_lm import stream_generate

        prompt = self.tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools) if tools else None,
            tokenize=False,
            add_generation_prompt=True,
        )

        with mlx_core.stream(mlx_core.new_thread_local_stream(mlx_core.gpu)):
            stream = stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
            )

            (id, _) = self._get_id_and_created()
            for index, chunk in enumerate(stream):
                yield self._create_chat_completion_chunk(chunk.text, index, id)

    def _get_id_and_created(self) -> tuple[str, int]:
        id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        return (id, created)

    def _create_chat_completion(
        self, content: str, id: str, created: int
    ) -> ChatCompletion:
        return ChatCompletion(
            id=id,
            created=created,
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content=content),
                )
            ],
            model=self.model_id,
            object="chat.completion",
        )

    def _create_chat_completion_chunk(
        self, content: str, index: int, id: str
    ) -> ChatCompletionChunk:

        created = int(time.time())

        return ChatCompletionChunk(
            id=id,
            created=created,
            choices=[
                ChunkChoice(
                    index=index, delta=ChoiceDelta(content=content, role="assistant")
                )
            ],
            model=self.model_id,
            object="chat.completion.chunk",
        )
