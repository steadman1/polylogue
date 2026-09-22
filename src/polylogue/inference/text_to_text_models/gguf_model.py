import gc
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, final

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from polylogue.inference.chat_templates.chat_template import (
    ChatTemplateConstants,
    ChatTemplateDetector,
)
from polylogue.inference.chat_templates.gguf_tokenizer_adapter import (
    GGUFTokenizerAdapter,
)

if TYPE_CHECKING:
    from llama_cpp import Llama


# Models should only be created using a factory
@final
class GGUFModel:
    def __init__(
        self,
        model_id: str,
        model_path: Path,
        n_ctx: int = 32_000,
        chat_format: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_path = model_path
        self.n_ctx: int = n_ctx
        self.chat_format = chat_format
        self.constants: ChatTemplateConstants | None = None

        self.model: Llama | None = None

    def load(self) -> None:
        from llama_cpp import Llama

        self.model = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            chat_format=self.chat_format,
            n_gpu_layers=-1,
            # flash_attn=True,
            # n_batch=512,
            # n_ubatch=256,
            type_k=1,  # Q8_0 or Q4_0 KV cache
            type_v=1,
            verbose=False,
        )

        metadata = self.model.metadata
        raw_template = metadata.get("tokenizer.chat_template", "")

        adapter = GGUFTokenizerAdapter(raw_template)
        self.constants = ChatTemplateDetector.from_tokenizer(adapter)

    def destroy(self) -> None:
        if not self.model:
            return

        self.model.close()
        self.model = None

        _ = gc.collect()

    def generate(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[str] | None = None,
    ) -> ChatCompletion:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        response = self.model.create_chat_completion(
            messages=list(messages),  # type: ignore
            tools=list(tools) if tools else None,  # type: ignore
            tool_choice="auto" if tools else None,
            stream=False,
            stop=["<|im_end|>", "</s>"],
        )

        return ChatCompletion.model_validate(response)

    def stream_generate(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None = None,
    ) -> Generator[ChatCompletionChunk, None, None]:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        stream = self.model.create_chat_completion(
            messages=list(messages),  # type: ignore
            tools=list(tools) if tools else None,  # type: ignore
            tool_choice="auto" if tools else None,
            stream=True,
        )

        yield from (ChatCompletionChunk.model_validate(chunk) for chunk in stream)
