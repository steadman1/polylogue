from __future__ import annotations

from collections.abc import Generator
from datetime import datetime
from typing import final

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice

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
        self, messages: list[dict[str, str | list[dict[str, str]]]]
    ) -> list[ChatCompletionMessageParam]:

        return list(MessageList(messages).clean())

    # generation should format the raw dictionary into an openai Completion
    def generate(
        self, messages: list[dict[str, str | list[dict[str, str]]]]
    ) -> ChatCompletion:
        timestamp_seconds = int(datetime.now().timestamp())

        messages: list[ChatCompletionMessageParam] = self.clean_messages(messages)
        response = self.model.generate(messages)

        choice = Choice(
            finish_reason="stop",
            index=0,
            message=ChatCompletionMessage(content=response, role="assistant"),
        )

        return ChatCompletion(
            id="0",
            choices=[choice],
            created=timestamp_seconds,
            model=self.model_id,
            object="chat.completion",
        )

    def stream_generate(
        self, messages: list[dict[str, str | list[dict[str, str]]]]
    ) -> Generator[str, None, None]:
        timestamp_seconds = int(datetime.now().timestamp())

        messages: list[ChatCompletionMessageParam] = self.clean_messages(messages)
        stream = self.model.stream_generate(messages)

        for is_last, chunk in generator_check_last(stream):
            choice = ChunkChoice(
                finish_reason="stop" if is_last else None,
                index=0,
                delta=ChoiceDelta(content=chunk, role="assistant"),
            )
            chunk = ChatCompletionChunk(
                id="0",
                choices=[choice],
                created=timestamp_seconds,
                model=self.model_id,
                object="chat.completion.chunk",
            )

            # Serialize Pydantic chunk to JSON string, formatted for SSE
            chunk_json = chunk.model_dump_json()
            yield f"data: {chunk_json}\n\n"
