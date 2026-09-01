import gc
from collections.abc import Generator
from pathlib import Path
from typing import final

from openai.types.chat import ChatCompletionMessageParam

from polylogue.inference.chat_templates.universal import CHATML_TEMPLATE


# Models should only be created usin a factory
@final
class MLXModel:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        from mlx_lm import load

        self.model, self.tokenizer = load(self.model_path)
        self.tokenizer.chat_template = CHATML_TEMPLATE

    def destroy(self) -> None:
        import mlx.core as mlx

        del self.model
        del self.tokenizer

        _ = gc.collect()
        mlx.clear_cache()

    def generate(self, messages: list[ChatCompletionMessageParam]) -> str:
        from mlx_lm import generate

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            max_tokens=128,
            add_generation_prompt=True,
            stop=["<|im_end|>", "</s>"],
        )

        return generate(self.model, self.tokenizer, prompt=prompt, verbose=True)

    def stream_generate(
        self, messages: list[ChatCompletionMessageParam]
    ) -> Generator[str, None, None]:
        from mlx_lm import stream_generate

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            max_tokens=128,
            add_generation_prompt=True,
            stop=["<|im_end|>", "</s>"],
        )

        stream = stream_generate(self.model, self.tokenizer, prompt, max_tokens=512)
        for chunk in stream:
            yield chunk.text
