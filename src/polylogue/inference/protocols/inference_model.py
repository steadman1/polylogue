from collections.abc import Generator, Sequence
from typing import Protocol, runtime_checkable

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)


@runtime_checkable
class InferenceModel(Protocol):
    def load(self) -> None: ...

    def destroy(self) -> None: ...

    def generate(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None,
    ) -> ChatCompletion: ...

    def stream_generate(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None,
    ) -> Generator[ChatCompletionChunk, None, None]: ...
