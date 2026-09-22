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
    # Universal / XML (Hermes, Qwen, ChatML)
    XML_TOOL_CALL_START = "<tool_call>"
    XML_TOOL_CALL_END = "</tool_call>"

    # Gemma Delimiters
    GEMMA_START_OF_TURN = "<start_of_turn>"
    GEMMA_END_OF_TURN = "<end_of_turn>"
    GEMMA_TOOL_CALL_START = "<|tool_call>"
    GEMMA_TOOL_CALL_END = "<tool_call|>"
    GEMMA_QUOTE_ESCAPE = '<|"|>'
    GEMMA_CALL_PREFIX = "call:"

    # Llama 3 Delimiters
    LLAMA_HEADER_START = "<|start_header_id|>"
    LLAMA_HEADER_END = "<|end_header_id|>"
    LLAMA_EOT_ID = "<|eot_id|>"
    LLAMA_EOM_ID = "<|eom_id|>"
    LLAMA_PYTHON_TAG = "<|python_tag|>"

    # ChatML Delimiters
    CHATML_IM_START = "<|im_start|>"
    CHATML_IM_END = "<|im_end|>"

    # Mistral Delimiters
    MISTRAL_TOOL_START = "[TOOL_CALLS]"
    MISTRAL_TOOL_END = "[/TOOL_CALLS]"

    # DeepSeek Delimiters
    DEEPSEEK_TOOL_START = "<｜tool call begin｜>"
    DEEPSEEK_TOOL_END = "<｜tool call end｜>"
    DEEPSEEK_TOOLS_WRAP_START = "<｜tool calls begin｜>"
    DEEPSEEK_TOOLS_WRAP_END = "<｜tool calls end｜>"
    DEEPSEEK_USER = "<｜User｜>"
    DEEPSEEK_ASSISTANT = "<｜Assistant｜>"
    DEEPSEEK_EOS = "<｜end of sentence｜>"

    # Thinking / Reasoning Delimiters
    THINK_START = "<think>"
    THINK_END = "</think>"
    THOUGHT_START = "<thought>"
    THOUGHT_END = "</thought>"


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
