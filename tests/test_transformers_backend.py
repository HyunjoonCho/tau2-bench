import pytest

from tau2.data_model.message import AssistantMessage, Message, SystemMessage, UserMessage
from tau2.environment.tool import Tool, as_tool
from tau2.llm.backends.transformers_backend import TransformersBackend


@pytest.fixture
def messages() -> list[Message]:
    return [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(role="user", content="hello"),
    ]


@pytest.fixture
def tool() -> Tool:
    def calculate_square(x: int) -> int:
        """Calculate a square."""
        return x * x

    return as_tool(calculate_square)


def test_transformers_backend_text_response(monkeypatch, messages: list[Message]):
    backend = TransformersBackend()

    monkeypatch.setattr(
        backend,
        "_generate_text",
        lambda **kwargs: (
            "This is a plain response",
            {"prompt_tokens": 10, "completion_tokens": 4},
            {"backend": "transformers"},
        ),
    )
    response = backend.generate(model="dummy-model", messages=messages)
    assert isinstance(response, AssistantMessage)
    assert response.content == "This is a plain response"
    assert response.tool_calls is None
    assert response.cost == 0.0
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 4}


def test_transformers_backend_tool_call_response(
    monkeypatch, messages: list[Message], tool: Tool
):
    backend = TransformersBackend()
    tool_call_json = '{"tool_calls":[{"name":"calculate_square","arguments":{"x":5}}]}'
    monkeypatch.setattr(
        backend,
        "_generate_text",
        lambda **kwargs: (
            tool_call_json,
            {"prompt_tokens": 12, "completion_tokens": 6},
            {"backend": "transformers"},
        ),
    )

    response = backend.generate(
        model="dummy-model",
        messages=messages,
        tools=[tool],
        tool_choice="required",
    )
    assert isinstance(response, AssistantMessage)
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "calculate_square"
    assert response.tool_calls[0].arguments == {"x": 5}


def test_transformers_backend_required_tool_call_enforced(
    monkeypatch, messages: list[Message], tool: Tool
):
    backend = TransformersBackend()
    monkeypatch.setattr(
        backend,
        "_generate_text",
        lambda **kwargs: (
            "I will answer without tools.",
            {"prompt_tokens": 8, "completion_tokens": 5},
            {"backend": "transformers"},
        ),
    )

    with pytest.raises(ValueError, match="tool_choice='required'"):
        backend.generate(
            model="dummy-model",
            messages=messages,
            tools=[tool],
            tool_choice="required",
        )


def test_generation_kwargs_skip_temperature_when_not_sampling():
    kwargs = TransformersBackend._build_generation_kwargs(
        tokenizer=type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 2})(),
        kwargs={"temperature": 0.0},
    )
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
