from collections.abc import Generator

from openai.types.chat import ChatCompletionMessageParam


class MockInferenceModel:
    def load(self) -> None:
        return

    def destroy(self) -> None:
        return

    def generate(self, messages: list[ChatCompletionMessageParam]) -> str:
        return "test non-streaming response"

    def stream_generate(
        self, messages: list[ChatCompletionMessageParam]
    ) -> Generator[str, None, None]:
        index, chunks = 0, ["test", "streaming", "response"]
        while index < len(chunks):
            yield chunks[index]
            index += 1
