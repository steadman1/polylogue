from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Generator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, final

from fastapi import Request
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

from polylogue.inference.chat_templates.chat_template import (
    ChatTemplateConstants,
    ChatTemplateDetector,
)
from polylogue.inference.helpers.message_list import MessageList
from polylogue.inference.helpers.stream_tool_buffer import StreamToolCallBuffer
from polylogue.inference.helpers.tool_call_parser import ToolCallParser
from polylogue.inference.protocols.inference_model import InferenceModel


@final
class TextToTextEngine:
    def __init__(self, model: InferenceModel, model_id: str) -> None:
        self.model = model
        self.model_id = model_id
        self._is_loaded = False
        self._load_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="engine-worker"
        )

    async def load(self) -> None:
        """Ensures the model and thread-local contexts are loaded on the executor."""
        if self._is_loaded:
            return

        async with self._load_lock:
            if not self._is_loaded:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._executor, self.model.load)
                self._is_loaded = True

        print(self.model.constants.tool_call_start, self.model.constants.tool_format)

    async def destroy(self) -> None:
        """Releases model parameters and halts the worker thread."""
        async with self._load_lock:
            if self._is_loaded:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._executor, self.model.destroy)
                self._is_loaded = False
        self._executor.shutdown(wait=False)

    async def generate(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None = None,
    ) -> ChatCompletion:
        await self.load()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._execute_generate, messages, tools
        )

    def _execute_generate(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None,
    ) -> ChatCompletion:
        cleaned = MessageList(messages).clean()
        response = self.model.generate(cleaned, tools)
        constants = getattr(self.model, "constants", None) or ChatTemplateConstants()

        choice = response.choices[0].message
        content = choice.content
        tool_calls: list[ChatCompletionMessageToolCall] = []

        # 1. Native engine tool calls
        if choice.tool_calls:
            tool_calls = [
                ChatCompletionMessageToolCall(
                    id=tc.id,
                    type="function",
                    function=Function(
                        name=tc.function.name,  # type: ignore
                        arguments=tc.function.arguments,  # type: ignore
                    ),
                )
                for tc in choice.tool_calls
            ]
        # 2. Text-formatted tool calls (XML, Gemma, Llama, Mistral)
        elif content and constants.tool_call_start in content:
            print(content)
            content, tool_calls = ToolCallParser.parse(content, constants.tool_format)

        return ChatCompletion(
            id=response.id or f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=response.created or int(time.time()),
            model=response.model or self.model_id,
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls or None,  # type: ignore
                    ),
                    finish_reason="tool_calls" if tool_calls else "stop",
                )
            ],
        )

    async def stream_generate(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None = None,
        request: Request | None = None,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        await self.load()

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[ChatCompletionChunk | Exception | object] = asyncio.Queue()
        stop_event = threading.Event()

        worker_future = loop.run_in_executor(
            self._executor,
            self._worker_stream_task,
            messages,
            tools,
            loop,
            queue,
            stop_event,
        )

        try:
            while True:
                if request is not None and await request.is_disconnected():
                    stop_event.set()
                    break

                item = await queue.get()
                if isinstance(item, Exception):
                    raise item
                if not isinstance(item, ChatCompletionChunk):
                    break

                yield item
                queue.task_done()
        finally:
            stop_event.set()
            await worker_future

    def _worker_stream_task(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Sequence[ChatCompletionToolParam] | None,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
        stop_event: threading.Event,
    ) -> None:
        sentinel = object()
        try:
            cleaned = MessageList(messages).clean()
            raw_stream = self.model.stream_generate(cleaned, tools)
            chunk_stream = self._transform_stream(raw_stream, tools, stop_event)

            for chunk in chunk_stream:
                # print(chunk.choices[0].finish_reason)
                if stop_event.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    def _transform_stream(
        self,
        stream: Generator[ChatCompletionChunk, None, None],
        tools: Sequence[ChatCompletionToolParam] | None,
        stop_event: threading.Event,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Transforms raw tokens into structured text chunks and tool call announcements."""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        constants = getattr(self.model, "constants", None) or ChatTemplateConstants()
        buffer = StreamToolCallBuffer(constants)

        for chunk in stream:
            if stop_event.is_set():
                return

            delta = chunk.choices[0].delta

            # print(delta.content, end="", flush=False)

            # 1. Native pass-through
            if delta.tool_calls:
                yield chunk
                continue

            token = delta.content or ""
            if delta.content is None:
                continue

            # 2. Passthrough if no tools registered
            if not tools:
                yield self._build_chunk(
                    completion_id, created, ChoiceDelta(content=token)
                )
                continue

            # 3. Process token via the tool buffer state machine
            for event in buffer.process_token(token):
                if isinstance(event, str):
                    yield self._build_chunk(
                        completion_id, created, ChoiceDelta(content=event)
                    )
                elif isinstance(event, list):
                    for call in event:
                        yield from self._yield_tool_call(call, completion_id, created)

        # 4. Emit terminal chunk
        for event in buffer.flush():
            if isinstance(event, str):
                yield self._build_chunk(
                    completion_id, created, ChoiceDelta(content=event)
                )
            elif isinstance(event, list):
                for call in event:
                    yield from self._yield_tool_call(call, completion_id, created)

    def _build_chunk(
        self,
        completion_id: str,
        created: int,
        delta: ChoiceDelta,
        finish_reason: str | None = None,
    ) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self.model_id,
            object="chat.completion.chunk",
            choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],  # type: ignore
        )

    def _yield_tool_call(
        self,
        call: ChatCompletionMessageToolCall,
        completion_id: str,
        created: int,
    ) -> Generator[ChatCompletionChunk, None, None]:
        # Header / Name chunk
        yield self._build_chunk(
            completion_id,
            created,
            ChoiceDelta(
                role="assistant",
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id=call.id,
                        type="function",
                        function=ChoiceDeltaToolCallFunction(
                            name=call.function.name, arguments=""
                        ),
                    )
                ],
            ),
        )

        # Body / Arguments chunk
        yield self._build_chunk(
            completion_id,
            created,
            ChoiceDelta(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        function=ChoiceDeltaToolCallFunction(
                            arguments=call.function.arguments
                        ),
                    )
                ]
            ),
        )

        # Terminal tool call finish reason chunk
        yield self._build_chunk(
            completion_id, created, ChoiceDelta(), finish_reason="tool_calls"
        )
