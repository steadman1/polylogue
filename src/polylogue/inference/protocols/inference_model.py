from collections.abc import Generator, Sequence
from typing import Protocol, runtime_checkable

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from polylogue.inference.chat_templates.chat_template import ChatTemplateConstants


@runtime_checkable
class InferenceModel(Protocol):
    constants: ChatTemplateConstants

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
