# TODO: make text_to_text_engine support dependency injection

from collections.abc import Generator
from types import GeneratorType

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from polylogue.constants import MOCK_MESSAGES
from polylogue.dependency_injection.mock_inference_model import MockInferenceModel
from polylogue.inference.protocols.inference_model import InferenceModel
from polylogue.inference.protocols.inference_model_engine import InferenceModelEngine
from polylogue.inference.text_to_text_engine import TextToTextEngine


def test_conforms_protocol_TextToTextEngine():
    model = MockInferenceModel()
    model_id = "test_model"

    assert isinstance(model, InferenceModel)

    engine = TextToTextEngine(model, model_id)

    assert isinstance(engine, InferenceModelEngine)


def test_TextToTextEngine() -> None:
    model = MockInferenceModel()
    model_id = "test_model"
    engine = TextToTextEngine(model, model_id)

    response: ChatCompletion = engine.generate(MOCK_MESSAGES)

    assert isinstance(response, ChatCompletion)
    message_content: str | None = response.choices[0].message.content
    assert message_content and len(message_content) > 0


def test_TextToTextEngine_streaming() -> None:
    model = MockInferenceModel()
    model_id = "test_model"
    engine = TextToTextEngine(model, model_id)

    stream: Generator[str, None, None] = engine.stream_generate(MOCK_MESSAGES)

    assert isinstance(stream, GeneratorType)

    for json_string in stream:
        # remove "data: " prefix, dont care abt the trailing "\n\n"
        raw_payload = json_string.removeprefix("data: ").strip()

        chunk = ChatCompletionChunk.model_validate_json(raw_payload)

        assert isinstance(chunk, ChatCompletionChunk)
        message_content: str | None = chunk.choices[0].delta.content
        assert message_content and len(message_content) > 0
