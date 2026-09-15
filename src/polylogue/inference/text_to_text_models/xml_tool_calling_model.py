from collections.abc import Generator

from openai.types.chat import (
    ChatCompletionChunk,
)
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
)


class XMLToolCallingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def _yield_token_as_plaintext(
        self, token: str, completion_id: str, created: int
    ) -> Generator[ChatCompletionChunk, None, None]:
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self.model_name,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(content=token),
                    finish_reason=None,
                )
            ],
        )

    def _yield_tool_call(
        self, call: ChatCompletionMessageToolCall, completion_id: str, created: int
    ) -> Generator[ChatCompletionChunk, None, None]:
        # announce tool call
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self.model_name,
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

        # yield arguments
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self.model_name,
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

        # terminate tool call
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self.model_name,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason="tool_calls",
                )
            ],
        )

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
