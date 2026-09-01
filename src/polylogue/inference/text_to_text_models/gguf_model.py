import gc
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, final

from openai.types.chat import ChatCompletionMessageParam

from polylogue.inference.helpers.message_list import MessageList

if TYPE_CHECKING:
    from llama_cpp import Llama


@final
class GGUFModel:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model: Llama | None = None

    def load(self) -> None:
        from llama_cpp import Llama

        self.model = Llama(
            model_path=str(self.model_path),
            n_ctx=0,
            n_gpu_layers=-1,
        )

    def destroy(self) -> None:
        if not self.model:
            return

        self.model.close()
        self.model = None

        _ = gc.collect()

    def generate(self, messages: list[ChatCompletionMessageParam]) -> str:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        messages = MessagesList(messages).clean()
        response = self.model.create_chat_completion(
            messages=messages,
            stream=False,
            stop=["<|im_end|>", "</s>"],
        )

        content = response["choices"][0]["message"]["content"]
        return content or ""

    def stream_generate(
        self, messages: list[ChatCompletionMessageParam]
    ) -> Generator[str, None, None]:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        messages = MessageList(messages).clean()
        stream = self.model.create_chat_completion(
            messages=messages,
            stream=True,
            stop=["<|im_end|>", "</s>"],
        )

        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content
