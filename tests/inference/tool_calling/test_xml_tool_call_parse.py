import json
from collections.abc import Generator

import pytest

from polylogue.inference.chat_templates.chat_constants import Delimiters, ToolCallFormat
from polylogue.inference.chat_templates.chat_template import ChatTemplateConstants
from polylogue.inference.helpers.stream_tool_buffer import StreamToolCallBuffer
from polylogue.inference.helpers.tool_call_parser import ToolCallParser

XML_TOOL_CALL = r"<tool_call><function=read><parameter=path>/path/to/something</parameter></function></tool_call>"

XML_CONSTANTS = ChatTemplateConstants(
    tool_call_start=Delimiters.XML_TOOL_CALL_START,
    tool_call_end=Delimiters.XML_TOOL_CALL_END,
    tool_format=ToolCallFormat.XML_HERMES,
)


def test_xml_non_streaming_tool_call_parse() -> None:
    raw_response = f"Making tool call. {XML_TOOL_CALL}"

    cleaned_text, tool_calls = ToolCallParser.parse(
        raw_response,
        fmt=ToolCallFormat.XML_HERMES,
    )

    # Verify tool envelope was stripped from body text
    assert cleaned_text == "Making tool call."
    assert len(tool_calls) == 1

    # Verify parsed function call and arguments
    call = tool_calls[0]
    assert call.function.name == "read"

    parsed_args = json.loads(call.function.arguments)
    assert parsed_args == {"path": "/path/to/something"}


def test_gemma_streaming_tool_call_parse() -> None:
    chunks = [
        XML_TOOL_CALL[:3],
        XML_TOOL_CALL[3:12],
        XML_TOOL_CALL[12:35],
        XML_TOOL_CALL[35:50],
        XML_TOOL_CALL[50:],
    ]

    def mock_gen() -> Generator[str, None, None]:
        yield from chunks

    buffer = StreamToolCallBuffer(XML_CONSTANTS)
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
    assert call.function.name == "read"
    assert json.loads(call.function.arguments) == {"path": "/path/to/something"}


def test_gemma_streaming_mixed_text_and_tool_call() -> None:
    stream_payload = [
        "Sure, ",
        "I can ",
        "check that for you.\n",
    ] + [
        XML_TOOL_CALL[:3],
        XML_TOOL_CALL[3:12],
        XML_TOOL_CALL[12:35],
        XML_TOOL_CALL[35:50],
        XML_TOOL_CALL[50:],
    ]

    buffer = StreamToolCallBuffer(XML_CONSTANTS)
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
    assert collected_calls[0].function.name == "read"
    assert json.loads(collected_calls[0].function.arguments) == {
        "path": "/path/to/something"
    }
