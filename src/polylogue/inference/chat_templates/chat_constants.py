import re
from enum import Enum


class ProbeMarkers:
    SYSTEM = "__SYS_PROBE__"
    USER = "__USER_PROBE__"
    ASSISTANT = "__ASST_PROBE__"
    USER_DUMMY_PROMPT = "hi"
    USER_DUMMY_TURN = "test"


class ToolCallFormat(str, Enum):
    XML_HERMES = "xml_hermes"  # <tool_call> ... </tool_call>
    GEMMA = "gemma"  # <|tool_call> ... <tool_call|>
    LLAMA = "llama"  # <|python_tag|> ... <|eom_id|>
    MISTRAL = "mistral"  # [TOOL_CALLS] ... [/TOOL_CALLS]
    DEEPSEEK = "deepseek"  # <｜tool call begin｜> ... <｜tool call end｜>
    UNKNOWN = "unknown"


class Delimiters:
    # Standard XML / Hermes
    XML_TOOL_CALL_START: str = "<tool_call>"
    XML_TOOL_CALL_END: str = "</tool_call>"
    XML_TOOL_RESPONSE_START: str = "<tool_response>"
    XML_TOOL_RESPONSE_END: str = "</tool_response>"

    # ChatML
    CHATML_IM_START: str = "<|im_start|>"
    CHATML_IM_END: str = "<|im_end|>"

    # Llama 3.x
    LLAMA_HEADER_START: str = "<|start_header_id|>"
    LLAMA_HEADER_END: str = "<|end_header_id|>"
    LLAMA_EOT_ID: str = "<|eot_id|>"
    LLAMA_EOM_ID: str = "<|eom_id|>"
    LLAMA_PYTHON_TAG: str = "<|python_tag|>"
    LLAMA_IPYTHON_START: str = "<|start_header_id|>ipython<|end_header_id|>\n\n"

    # Gemma / Gemma 4
    GEMMA_START_OF_TURN: str = "<start_of_turn>"
    GEMMA_END_OF_TURN: str = "<end_of_turn>"
    GEMMA_CALL_PREFIX: str = "call:"
    GEMMA_QUOTE_ESCAPE: str = '<|\\"|>'
    GEMMA_TOOL_CALL_START: str = "<|tool_call>"
    GEMMA_TOOL_CALL_END: str = "<tool_call|>"
    GEMMA_TOOL_RESPONSE_START: str = "<|tool_response>"
    GEMMA_TOOL_RESPONSE_END: str = "<tool_response|>"
    GEMMA_CHANNEL_START: str = "<|channel>"
    GEMMA_CHANNEL_END: str = "<channel|>"
    GEMMA_THINK_START: str = "<|channel>thought\n"
    GEMMA_THINK_END: str = "<channel|>"

    # Mistral / Mixtral
    MISTRAL_TOOL_START: str = "[TOOL_CALLS]"
    MISTRAL_TOOL_END: str = ""
    MISTRAL_TOOL_RESPONSE_START: str = "[TOOL_RESULTS]"
    MISTRAL_TOOL_RESPONSE_END: str = "[/TOOL_RESULTS]"

    # DeepSeek
    DEEPSEEK_USER: str = "<｜User｜>"
    DEEPSEEK_ASSISTANT: str = "<｜Assistant｜>"
    DEEPSEEK_EOS: str = "<｜end of sentence｜>"
    DEEPSEEK_TOOL_START: str = "<｜tool calls｜>"
    DEEPSEEK_TOOL_END: str = "<｜end of tool calls｜>"
    DEEPSEEK_TOOLS_WRAP_START: str = "<｜tool call begin｜>"
    DEEPSEEK_TOOL_RESPONSE_START: str = "<｜tool outputs｜>"
    DEEPSEEK_TOOL_RESPONSE_END: str = "<｜end of tool outputs｜>"

    # Generic Reasoning Delimiters
    THINK_START: str = "<think>"
    THINK_END: str = "</think>"
    THOUGHT_START: str = "<thought>"
    THOUGHT_END: str = "</thought>"


class Patterns:
    # Reasoning tag discovery
    THINK_OPENING = re.compile(r"(<think>|\[THINK\]|<thought>|<reasoning>)")
    THINK_CLOSING = re.compile(r"(</think>|\[/THINK\]|</thought>|</reasoning>)")

    # Tool call extraction patterns
    XML_ENVELOPE = re.compile(
        rf"{re.escape(Delimiters.XML_TOOL_CALL_START)}(.*?){re.escape(Delimiters.XML_TOOL_CALL_END)}",
        re.DOTALL,
    )
    GEMMA_ENVELOPE = re.compile(
        rf"{re.escape(Delimiters.GEMMA_TOOL_CALL_START)}(.*?){re.escape(Delimiters.GEMMA_TOOL_CALL_END)}",
        re.DOTALL,
    )
    LLAMA_ENVELOPE = re.compile(
        rf"{re.escape(Delimiters.LLAMA_PYTHON_TAG)}(.*?)(?:{re.escape(Delimiters.LLAMA_EOM_ID)}|$)",
        re.DOTALL,
    )
    MISTRAL_ENVELOPE = re.compile(
        rf"{re.escape(Delimiters.MISTRAL_TOOL_START)}\s*(\[.*?\]|{{.*?}})(?:{re.escape(Delimiters.MISTRAL_TOOL_END)}|$)",
        re.DOTALL,
    )

    # Gemma call syntax: call:fn_name{...}
    GEMMA_CALL = re.compile(r"call:([a-zA-Z0-9_\-\.]+)\s*(\{.*\})", re.DOTALL)
    JSON_UNQUOTED_KEY = re.compile(r"([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", re.DOTALL)

    # XML inner parameter extraction
    XML_INNER_NAME = re.compile(r"<name>(.*?)</name>", re.DOTALL)
    XML_INNER_ARGS = re.compile(r"<arguments>(.*?)</arguments>", re.DOTALL)
