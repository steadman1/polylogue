from __future__ import annotations

from collections.abc import Generator, Sequence
from datetime import datetime
from typing import final

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from polylogue.helpers.generator_check_last import generator_check_last
from polylogue.inference.helpers.message_list import MessageList
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

        return self.model.generate(cleaned_messages, tools)

    def stream_generate(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None = None,
    ) -> Generator[str, None, None]:
        cleaned_messages: list[ChatCompletionMessageParam] = self.clean_messages(
            messages
        )

        stream = self.model.stream_generate(cleaned_messages, tools)

        for chunk in stream:
            # Serialize Pydantic chunk to JSON string, formatted for SSE
            chunk_json = chunk.model_dump_json()
            yield f"data: {chunk_json}\n\n"
