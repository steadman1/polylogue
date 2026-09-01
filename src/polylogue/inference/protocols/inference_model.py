from collections.abc import Generator
from typing import Protocol, runtime_checkable

from openai.types.chat import ChatCompletionMessageParam


@runtime_checkable
class InferenceModel(Protocol):
    def load(self) -> None: ...

    def destroy(self) -> None: ...

    def generate(self, messages: list[ChatCompletionMessageParam]) -> str: ...

    def stream_generate(
        self, messages: list[ChatCompletionMessageParam]
    ) -> Generator[str, None, None]: ...
