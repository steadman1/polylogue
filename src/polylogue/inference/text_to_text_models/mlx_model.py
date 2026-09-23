import gc
import time
import uuid
from collections.abc import Generator, Sequence
from pathlib import Path
from types import ModuleType
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

from polylogue.inference.chat_templates.chat_template import (
    ChatTemplateConstants,
    ChatTemplateDetector,
)


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
        self.constants: ChatTemplateConstants | None = None

        self.model: Any = None
        self.tokenizer: Any = None
        self._mlx_core: ModuleType | None = None

    @property
    def _mx(self) -> ModuleType:
        if self._mlx_core is None:
            import mlx.core as mx

            self._mlx_core = mx
        return self._mlx_core

    def load(self) -> None:
        from mlx_lm import load

        mx = self._mx
        mx.set_default_device(mx.gpu)

        self.model, self.tokenizer = load(str(self.model_path))  # type: ignore
        self.constants = ChatTemplateDetector.from_model_dir(self.model_path)

        mx.eval(self.model.parameters())
        mx.synchronize()

    def destroy(self) -> None:
        self.model = None
        self.tokenizer = None

        _ = gc.collect()
        if self._mlx_core is not None:
            self._mlx_core.metal.clear_cache()

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

        # Bind the stream to the current thread's GPU stream
        mx = self._mx
        with mx.stream(mx.default_stream(mx.gpu)):
            response: str = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                verbose=False,
            )

        id_str, created_time = self._get_id_and_created()
        return self._create_chat_completion(response, id_str, created_time)

    def stream_generate(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None = None,
    ) -> Generator[ChatCompletionChunk, None, None]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache

        mx = self._mx
        mx.set_default_device(mx.gpu)

        prompt = self.tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools) if tools else None,
            tokenize=False,
            add_generation_prompt=True,
        )

        id_str, _ = self._get_id_and_created()

        print("WHAT THE MODELS SEES ======== " + prompt)

        prompt_cache = make_prompt_cache(self.model)
        stream = stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            prompt_cache=prompt_cache,
        )

        for index, chunk in enumerate(stream):
            yield self._create_chat_completion_chunk(chunk.text, index, id_str)

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
