import gc
import json
import re
import time
import uuid
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, final

from llama_cpp import Llama
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
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from polylogue.constants import XML_TOOL_CALL_END, XML_TOOL_CALL_START
from polylogue.inference.helpers.parse_xml_tool_calls import parse_xml_tool_calls
from polylogue.inference.text_to_text_models.xml_tool_calling_model import (
    XMLToolCallingModel,
)

if TYPE_CHECKING:
    from llama_cpp import Llama


@final
class GGUFModel(XMLToolCallingModel):
    def __init__(
        self,
        model_name: str,
        model_path: Path,
        n_ctx: int = 128_000,
        chat_format: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.n_ctx: int = n_ctx
        self.chat_format = chat_format

        self.model: Llama | None = None

    def load(self) -> None:
        from llama_cpp import Llama

        self.model = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            chat_format=self.chat_format,
            n_gpu_layers=-1,
            type_k=1,  # Q8_0 or Q4_0 KV cache
            type_v=1,
        )

    def destroy(self) -> None:
        if not self.model:
            return

        self.model.close()
        self.model = None

        _ = gc.collect()

    def generate(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None = None,
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

        choice_dict = response["choices"][0]["message"]  # type: ignore
        content = choice_dict.get("content") or ""
        tool_calls: Sequence[ChatCompletionMessageToolCall] | None = None

        if "tool_calls" in choice_dict and choice_dict["tool_calls"]:
            tool_calls = [
                ChatCompletionMessageToolCall(
                    id=tool_call["id"],
                    type="function",
                    function=Function(
                        name=tool_call["function"]["name"],
                        arguments=tool_call["function"]["arguments"],
                    ),
                )
                for tool_call in choice_dict["tool_calls"]
            ]
        elif "<tool_call>" in content:
            content, tool_calls = parse_xml_tool_calls(content)

        return ChatCompletion(
            id=response.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),  # type: ignore
            created=response.get("created", int(time.time())),  # type: ignore
            model=response.get("model", "gguf-model"),  # type: ignore
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=choice_dict.get("content"),
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
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        stream = self.model.create_chat_completion(
            messages=list(messages),  # type: ignore
            tools=list(tools) if tools else None,  # type: ignore
            tool_choice="auto" if tools else None,
            stream=True,
        )

        tool_call_buffer = ""
        tool_call_full_content = ""
        tool_call_active = False
        tool_call_emitted = False

        for chunk in stream:
            choice = chunk["choices"][0]  # type: ignore
            delta = choice.get("delta", {})  # type: ignore

            # 1. Native llama-cpp handled tool calls pass-through
            if delta.get("tool_calls"):
                tool_call_emitted = True
                yield ChatCompletionChunk.model_validate(chunk)
                continue

            token: str = delta.get("content") or ""
            if not token:
                continue

            # 2. Plain completion passthrough if no tools were supplied
            if not tools:
                yield from self._yield_token_as_plaintext(token, completion_id, created)
                continue

            new_token_buffer = tool_call_buffer + token

            # 3. Look for tool call opening tag
            if not tool_call_active:
                match = self._message_check_contents_for_target_fragment(
                    XML_TOOL_CALL_START, new_token_buffer
                )
                if match:
                    start_idx, end_idx = match
                    # Yield any text that appeared BEFORE the tool call tag
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
                        # Keep holding the partial prefix (e.g. "<tool")
                        tool_call_buffer = matched_fragment
                else:
                    # No tag match or prefix found: flush entire buffer as plaintext
                    yield from self._yield_token_as_plaintext(
                        new_token_buffer, completion_id, created
                    )
                    tool_call_buffer = ""

            # 4. In active tool call: look for closing tag
            else:
                match = self._message_check_contents_for_target_fragment(
                    XML_TOOL_CALL_END, new_token_buffer
                )
                if match:
                    start_idx, end_idx = match
                    matched_fragment = new_token_buffer[start_idx:end_idx]

                    if matched_fragment == XML_TOOL_CALL_END:
                        # Append parameter content up to </tool_call>
                        tool_call_full_content += new_token_buffer[:start_idx]

                        _, tool_calls = parse_xml_tool_calls(
                            f"{XML_TOOL_CALL_START}{tool_call_full_content}{XML_TOOL_CALL_END}"
                        )
                        if tool_calls:
                            tool_call_emitted = True
                            yield from self._yield_tool_call(
                                tool_calls[0], completion_id, created
                            )

                        tool_call_full_content = ""
                        tool_call_buffer = ""
                        tool_call_active = False
                    else:
                        # Incomplete closing tag fragment: buffer it
                        tool_call_full_content += new_token_buffer[:start_idx]
                        tool_call_buffer = matched_fragment
                else:
                    # Still inside parameters
                    tool_call_full_content += new_token_buffer
                    tool_call_buffer = ""

        # 5. Flush any trailing buffered characters that never became tags
        trailing = tool_call_buffer or (
            tool_call_full_content if not tool_call_emitted else ""
        )
        if trailing and not tool_call_emitted:
            yield from self._yield_token_as_plaintext(trailing, completion_id, created)

        # 6. Emit terminal chunk ONLY if finish_reason="tool_calls" was not already sent
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
