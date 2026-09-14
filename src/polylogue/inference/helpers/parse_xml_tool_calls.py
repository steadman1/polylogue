import json
import re
import uuid

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)


def parse_xml_tool_calls(
    raw_content: str,
) -> tuple[str | None, list[ChatCompletionMessageToolCall] | None]:
    """Parses XML tool call formats like <tool_call><function=name><parameter=k>v</parameter></function></tool_call>"""
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    matches = list(re.finditer(tool_call_pattern, raw_content, re.DOTALL))

    if not matches:
        return raw_content, None

    parsed_calls: list[ChatCompletionMessageToolCall] = []

    for match in matches:
        block = match.group(1).strip()

        # Extract function name from <function=NAME>...</function>
        fn_match = re.search(
            r"<function=([a-zA-Z0-9_\-]+)>(.*?)</function>", block, re.DOTALL
        )
        if not fn_match:
            continue

        fn_name = fn_match.group(1).strip()
        params_block = fn_match.group(2)

        # Extract parameters: <parameter=KEY>VALUE</parameter>
        param_matches = re.findall(
            r"<parameter=([a-zA-Z0-9_\-]+)>\s*(.*?)\s*</parameter>",
            params_block,
            re.DOTALL,
        )

        arguments_dict: dict[str, str] = {}
        for k, v in param_matches:
            cleaned_val = v.strip()
            # Attempt to parse json scalars (numbers, booleans, sub-json)
            try:
                arguments_dict[k] = json.loads(cleaned_val)
            except (json.JSONDecodeError, ValueError):
                arguments_dict[k] = cleaned_val

        parsed_calls.append(
            ChatCompletionMessageToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=Function(
                    name=fn_name,
                    arguments=json.dumps(arguments_dict),
                ),
            )
        )

    # Strip tool call tags out of remaining text
    cleaned_content = re.sub(
        tool_call_pattern, "", raw_content, flags=re.DOTALL
    ).strip()
    return (cleaned_content if cleaned_content else None), (
        parsed_calls if parsed_calls else None
    )
