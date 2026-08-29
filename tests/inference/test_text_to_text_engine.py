# TODO: make text_to_text_engine support dependency injection

from collections.abc import Generator
from types import GeneratorType

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from polylogue.dependency_injection.test_inference_model import TestInferenceModel
from polylogue.inference.text_to_text_engine import TextToTextEngine


def test_TextToTextEngine() -> None:
    model = TestInferenceModel()
    model_id = "test_model"
    engine = TextToTextEngine(model, model_id)

    response: ChatCompletion = engine.generate("prompt")

    assert isinstance(response, ChatCompletion)
    message_content: str | None = response.choices[0].message.content
    assert message_content and len(message_content) > 0


def test_TextToTextEngine_streaming() -> None:
    model = TestInferenceModel()
    model_id = "test_model"
    engine = TextToTextEngine(model, model_id)

    stream: Generator[str, None, None] = engine.stream_generate("prompt")

    assert isinstance(stream, GeneratorType)

    for json_string in stream:
        # remove "data: " prefix, dont care abt the trailing "\n\n"
        raw_payload = json_string.removeprefix("data: ").strip()

        chunk = ChatCompletionChunk.model_validate_json(raw_payload)

        assert isinstance(chunk, ChatCompletionChunk)
        message_content: str | None = chunk.choices[0].delta.content
        assert message_content and len(message_content) > 0
