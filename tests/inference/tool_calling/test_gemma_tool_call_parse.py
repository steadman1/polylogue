import json
from collections.abc import Generator

import pytest

from polylogue.inference.chat_templates.chat_constants import Delimiters, ToolCallFormat
from polylogue.inference.chat_templates.chat_template import ChatTemplateConstants
from polylogue.inference.helpers.stream_tool_buffer import StreamToolCallBuffer
from polylogue.inference.helpers.tool_call_parser import ToolCallParser

GEMMA_TOOL_CALL = (
    r'<|tool_call>call:get_current_temperature{location:<|"|>London<|"|>}<tool_call|>'
)

GEMMA_CONSTANTS = ChatTemplateConstants(
    tool_call_start=Delimiters.GEMMA_TOOL_CALL_START,
    tool_call_end=Delimiters.GEMMA_TOOL_CALL_END,
    tool_format=ToolCallFormat.GEMMA,
)


def test_gemma_non_streaming_tool_call_parse() -> None:
    raw_response = f"Checking the weather now. {GEMMA_TOOL_CALL}"

    cleaned_text, tool_calls = ToolCallParser.parse(
        raw_response,
        fmt=ToolCallFormat.GEMMA,
    )

    # Verify tool envelope was stripped from body text
    assert cleaned_text == "Checking the weather now."
    assert len(tool_calls) == 1

    # Verify parsed function call and arguments
    call = tool_calls[0]
    assert call.function.name == "get_current_temperature"

    parsed_args = json.loads(call.function.arguments)
    assert parsed_args == {"location": "London"}


def test_gemma_streaming_tool_call_parse() -> None:
    # Arbitrary chunk slices slicing mid-token and across tags
    chunks = [
        GEMMA_TOOL_CALL[:3],  # '<|t'
        GEMMA_TOOL_CALL[3:12],  # 'ool_call>c'
        GEMMA_TOOL_CALL[12:35],  # 'all:get_current_tempera'
        GEMMA_TOOL_CALL[35:60],  # 'ture{location:<|"|>London'
        GEMMA_TOOL_CALL[60:],  # '<|"|>}<tool_call|>'
    ]

    def mock_gen() -> Generator[str, None, None]:
        for chunk in chunks:
            yield chunk

    buffer = StreamToolCallBuffer(GEMMA_CONSTANTS)
    emitted_text = []
    emitted_calls = []

    for chunk in mock_gen():
        for event in buffer.process_token(chunk):
            if isinstance(event, str):
                emitted_text.append(event)
            elif isinstance(event, list):
                emitted_calls.extend(event)

    for trailing in buffer.flush():
        emitted_text.append(trailing)

    # All text inside <|tool_call>...<tool_call|> should be absorbed
    assert "".join(emitted_text) == ""
    assert len(emitted_calls) == 1

    call = emitted_calls[0]
    assert call.function.name == "get_current_temperature"
    assert json.loads(call.function.arguments) == {"location": "London"}


def test_gemma_streaming_mixed_text_and_tool_call() -> None:
    stream_payload = [
        "Sure, ",
        "I can ",
        "check that for you.\n",
        "<|tool_call",
        '>call:get_current_temperature{location:<|"',
        '|>London<|"|',
        ">}<tool_call",
        "|>",
    ]

    buffer = StreamToolCallBuffer(GEMMA_CONSTANTS)
    collected_prose = []
    collected_calls = []

    for token in stream_payload:
        for item in buffer.process_token(token):
            if isinstance(item, str):
                collected_prose.append(item)
            elif isinstance(item, list):
                collected_calls.extend(item)

    for trailing in buffer.flush():
        collected_prose.append(trailing)

    assert "".join(collected_prose) == "Sure, I can check that for you.\n"
    assert len(collected_calls) == 1
    assert collected_calls[0].function.name == "get_current_temperature"
    assert json.loads(collected_calls[0].function.arguments) == {"location": "London"}


def test_gemma_multiple_and_typed_arguments() -> None:
    raw_call = (
        r"<|tool_call>call:set_alarm{"
        r'label:<|"|>Morning<|"|>, '
        r'time:<|"|>07:00<|"|>, '
        r"repeat:true, "
        r"snooze_duration:10"
        r"}<tool_call|>"
    )

    cleaned_text, tool_calls = ToolCallParser.parse(
        raw_call,
        fmt=ToolCallFormat.GEMMA,
    )

    assert cleaned_text == ""
    assert len(tool_calls) == 1

    call = tool_calls[0]
    assert call.function.name == "set_alarm"

    args = json.loads(call.function.arguments)
    assert args == {
        "label": "Morning",
        "time": "07:00",
        "repeat": True,
        "snooze_duration": 10,
    }
