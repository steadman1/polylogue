from collections.abc import Generator


class MockInferenceModel:
    def load(self) -> None:
        return

    def destroy(self) -> None:
        return

    def generate(self, prompt: str) -> str:
        _ = prompt
        return "test non-streaming response"

    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        _ = prompt

        index, chunks = 0, ["test", "streaming", "response"]
        while index < len(chunks):
            yield chunks[index]
            index += 1
