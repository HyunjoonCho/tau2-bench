import json

import pytest

from tau2.data_model.message import AssistantMessage, Message, SystemMessage, UserMessage
from tau2.llm.backends.litellm_backend import LiteLLMBackend
from tau2.utils import llm_utils


@pytest.fixture
def messages() -> list[Message]:
    return [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(role="user", content="Hi"),
    ]


def test_generate_dispatches_default_backend(monkeypatch, messages: list[Message]):
    class DummyBackend:
        def __init__(self):
            self.called = False

        def generate(self, **kwargs):
            self.called = True
            return AssistantMessage(role="assistant", content="ok")

    backend = DummyBackend()
    monkeypatch.setitem(llm_utils._BACKENDS, "litellm", backend)
    response = llm_utils.generate(model="dummy", messages=messages)
    assert isinstance(response, AssistantMessage)
    assert response.content == "ok"
    assert backend.called


def test_generate_unknown_backend_raises(messages: list[Message]):
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        llm_utils.generate(model="dummy", messages=messages, llm_backend="not-a-backend")


def test_litellm_backend_parses_tool_calls(monkeypatch, messages: list[Message]):
    backend = LiteLLMBackend()

    class FakeUsage:
        completion_tokens = 8
        prompt_tokens = 10

    class FakeFunction:
        name = "calculate_square"
        arguments = json.dumps({"x": 5})

    class FakeToolCall:
        id = "call_1"
        function = FakeFunction()

    class FakeMessage:
        role = "assistant"
        content = None
        tool_calls = [FakeToolCall()]

    class FakeChoice:
        finish_reason = "stop"
        message = FakeMessage()

        @staticmethod
        def to_dict():
            return {"raw": "choice"}

    class FakeResponse:
        model = "gpt-4.1"
        choices = [FakeChoice()]

        @staticmethod
        def get(key):
            if key == "usage":
                return FakeUsage()
            return None

    def fake_completion(**kwargs):
        return FakeResponse()

    monkeypatch.setattr("tau2.llm.backends.litellm_backend.completion", fake_completion)
    monkeypatch.setattr(
        "tau2.llm.backends.litellm_backend.completion_cost",
        lambda completion_response: 0.123,
    )

    response = backend.generate(model="gpt-4.1", messages=messages)
    assert isinstance(response, AssistantMessage)
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "calculate_square"
    assert response.tool_calls[0].arguments == {"x": 5}
    assert response.cost == 0.123
    assert response.usage == {"completion_tokens": 8, "prompt_tokens": 10}


def test_litellm_backend_sets_default_num_retries(monkeypatch, messages: list[Message]):
    backend = LiteLLMBackend()
    captured = {}

    class FakeMessage:
        role = "assistant"
        content = "hello"
        tool_calls = None

    class FakeChoice:
        finish_reason = "stop"
        message = FakeMessage()

        @staticmethod
        def to_dict():
            return {"raw": "choice"}

    class FakeResponse:
        model = "gpt-4.1"
        choices = [FakeChoice()]

        @staticmethod
        def get(key):
            return None

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("tau2.llm.backends.litellm_backend.completion", fake_completion)
    monkeypatch.setattr(
        "tau2.llm.backends.litellm_backend.completion_cost",
        lambda completion_response: 0.0,
    )

    response = backend.generate(model="gpt-4.1", messages=messages)
    assert isinstance(response, AssistantMessage)
    assert "num_retries" in captured
    assert captured["num_retries"] == 3
