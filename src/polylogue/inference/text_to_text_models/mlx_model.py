import gc
import time
import uuid
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, final

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from polylogue.constants import XML_TOOL_CALL_END, XML_TOOL_CALL_START
from polylogue.inference.chat_templates.universal import CHATML_TEMPLATE
from polylogue.inference.helpers.parse_xml_tool_calls import parse_xml_tool_calls
from polylogue.inference.text_to_text_models.xml_tool_calling_model import (
    XMLToolCallingModel,
)


# Models should only be created usin a factory
@final
class MLXModel(XMLToolCallingModel):
    def __init__(
        self,
        model_name: str,
        model_path: Path,
        max_tokens: int = 1,
    ) -> None:
        self.model_name = model_name
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

        raw_output: str = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            verbose=False,
        )

        content: str = raw_output
        tool_calls: Sequence[ChatCompletionMessageToolCall] | None = None

        if XML_TOOL_CALL_START in raw_output:
            cleaned_content, parsed_tool_calls = parse_xml_tool_calls(raw_output)
            content = cleaned_content if cleaned_content is not None else ""
            tool_calls = parsed_tool_calls

        return ChatCompletion(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=self.model_name,
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=content or None,
                        tool_calls=list(tool_calls) if tool_calls else [],
                    ),
                    finish_reason="tool_calls" if tool_calls else "stop",
                )
            ],
        )

    def stream_generate(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None = None,
    ) -> Generator[ChatCompletionChunk, None, None]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        from mlx_lm import stream_generate

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        prompt = self.tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools) if tools else None,
            tokenize=False,
            add_generation_prompt=True,
        )

        stream = stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
        )

        tool_call_buffer = ""
        tool_call_full_content = ""
        tool_call_active = False
        tool_call_emitted = False

        for chunk in stream:
            token: str = getattr(chunk, "text", "")
            if not token:
                continue

            # 1. Plain completion passthrough if no tools were supplied
            if not tools:
                yield from self._yield_token_as_plaintext(token, completion_id, created)
                continue

            new_token_buffer = tool_call_buffer + token

            # 2. Look for tool call opening tag
            if not tool_call_active:
                match = self._message_check_contents_for_target_fragment(
                    XML_TOOL_CALL_START, new_token_buffer
                )
                if match:
                    start_idx, end_idx = match
                    pre_text = new_token_buffer[:start_idx]
                    if pre_text:
                        yield from self._yield_token_as_plaintext(
                            pre_text, completion_id, created
                        )

                    matched_fragment = new_token_buffer[start_idx:end_idx]
                    if matched_fragment == XML_TOOL_CALL_START:
                        tool_call_active = True
                        tool_call_buffer = ""
                        tool_call_full_content = ""
                    else:
                        tool_call_buffer = matched_fragment
                else:
                    yield from self._yield_token_as_plaintext(
                        new_token_buffer, completion_id, created
                    )
                    tool_call_buffer = ""

            # 3. Active tool call: capture arguments and seek closing tag
            else:
                match = self._message_check_contents_for_target_fragment(
                    XML_TOOL_CALL_END, new_token_buffer
                )
                if match:
                    start_idx, end_idx = match
                    matched_fragment = new_token_buffer[start_idx:end_idx]

                    if matched_fragment == XML_TOOL_CALL_END:
                        tool_call_full_content += new_token_buffer[:start_idx]

                        _, parsed_calls = parse_xml_tool_calls(
                            f"{XML_TOOL_CALL_START}{tool_call_full_content}{XML_TOOL_CALL_END}"
                        )
                        if parsed_calls:
                            tool_call_emitted = True
                            yield from self._yield_tool_call(
                                parsed_calls[0], completion_id, created
                            )

                        tool_call_full_content = ""
                        tool_call_buffer = ""
                        tool_call_active = False
                    else:
                        tool_call_full_content += new_token_buffer[:start_idx]
                        tool_call_buffer = matched_fragment
                else:
                    tool_call_full_content += new_token_buffer
                    tool_call_buffer = ""

        # 4. Flush trailing characters that did not form complete tags
        trailing = tool_call_buffer or (
            tool_call_full_content if not tool_call_emitted else ""
        )
        if trailing and not tool_call_emitted:
            yield from self._yield_token_as_plaintext(trailing, completion_id, created)

        # 5. Emit terminal chunk if finish_reason="tool_calls" was not sent
        if not tool_call_emitted:
            yield ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=self.model_name,
                object="chat.completion.chunk",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(),
                        finish_reason="stop",
                    )
                ],
            )
