from __future__ import annotations

import time
import uuid
from collections.abc import Generator, Sequence
from typing import final

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
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
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from polylogue.constants import XML_TOOL_CALL_END, XML_TOOL_CALL_START
from polylogue.inference.helpers.message_list import MessageList
from polylogue.inference.helpers.parse_xml_tool_calls import (
    parse_xml_tool_calls,
)
from polylogue.inference.protocols.inference_model import InferenceModel


@final
class TextToTextEngine:
    def __init__(self, model: InferenceModel, model_id: str):
        # to support dependecy injection, we need to take in an object
        # that will handle choosing the model
        self.model: InferenceModel = model
        self.model_id: str = model_id

        self.model.load()

    def destroy(self) -> None:
        self.model.destroy()

    def clean_messages(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:

        return MessageList(messages).clean()

    # generation should format the raw dictionary into an openai Completion
    def generate(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None = None,
    ) -> ChatCompletion:
        cleaned_messages: list[ChatCompletionMessageParam] = self.clean_messages(
            messages
        )

        response = self.model.generate(cleaned_messages, tools)

        choice_dict = response["choices"][0]["message"]  # type: ignore
        content = choice_dict.get("content") or ""
        tool_calls: Sequence[ChatCompletionMessageToolCall] | None = None

        if choice_dict.get("tool_calls"):
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
    ) -> Generator[str, None, None]:
        cleaned_messages: list[ChatCompletionMessageParam] = self.clean_messages(
            messages
        )

        stream = self.model.stream_generate(cleaned_messages, tools)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        tool_call_buffer = ""
        tool_call_full_content = ""
        tool_call_active = False
        tool_call_emitted = False

        for chunk in stream:
            choice = chunk.choices[0]  # type: ignore
            delta = choice.delta  # type: ignore

            # 1. Native llama-cpp handled tool calls pass-through
            if delta.tool_calls:
                tool_call_emitted = True
                yield self._serialize_chunk(ChatCompletionChunk.model_validate(chunk))
                continue

            token: str = delta.content or ""
            if not token:
                continue

            # 2. Plain completion passthrough if no tools were supplied
            if not tools:
                yield self._return_chunk_as_plain_text(token, completion_id, created)
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
                        yield from self._return_chunk_as_plain_text(
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
                    yield self._return_chunk_as_plain_text(
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
            yield self._return_chunk_as_plain_text(trailing, completion_id, created)

        # 6. Emit terminal chunk ONLY if finish_reason="tool_calls" was not already sent
        if not tool_call_emitted:
            yield self._serialize_chunk(
                ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=self.model_id,
                    object="chat.completion.chunk",
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=ChoiceDelta(),
                            finish_reason="stop",
                        )
                    ],
                )
            )

    def _serialize_chunk(self, chunk: ChatCompletionChunk) -> str:
        # Serialize Pydantic chunk to JSON string, formatted for SSE
        chunk_json = chunk.model_dump_json()
        return f"data: {chunk_json}\n\n"

    def _return_chunk_as_plain_text(
        self, token: str, completion_id: str, created: int
    ) -> str:

        plain_text_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self.model_id,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(content=token),
                    finish_reason=None,
                )
            ],
        )

        return self._serialize_chunk(plain_text_chunk)

    def _yield_tool_call(
        self,
        call: ChatCompletionMessageToolCall,
        completion_id: str,
        created: int,
    ) -> Generator[str, None, None]:
        tool_call_announcement = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self.model_id,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id=call.id,
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name=call.function.name,
                                    arguments="",
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        )
        tool_call_arguments = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self.model_id,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                function=ChoiceDeltaToolCallFunction(
                                    arguments=call.function.arguments
                                ),
                            )
                        ]
                    ),
                    finish_reason=None,
                )
            ],
        )
        terminate_tool_call = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self.model_id,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason="tool_calls",
                )
            ],
        )
        yield self._serialize_chunk(tool_call_announcement)

        yield self._serialize_chunk(tool_call_arguments)

        yield self._serialize_chunk(terminate_tool_call)

    def _message_check_contents_for_target_fragment(
        self, target: str, message: str
    ) -> tuple[int, int] | None:
        if not message or not target:
            return None

        # Step 1: Check for complete occurrence anywhere in message
        target_len = len(target)
        for i in range(len(message) - target_len + 1):
            if message[i : i + target_len] == target:
                return (i, i + target_len)

        # Step 2: Check if the tail of message matches a prefix of target (no chars after)
        max_overlap = min(len(message), target_len - 1)
        for length in range(max_overlap, 0, -1):
            start = len(message) - length
            if message[start:] == target[:length]:
                return (start, len(message))

        return None
