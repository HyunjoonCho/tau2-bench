from tau2.data_model.message import AssistantMessage, Message, SystemMessage, UserMessage
from tau2.llm.backends.guidance_backend import GuidanceBackend


def test_guidance_backend_delegates_to_transformers(monkeypatch):
    backend = GuidanceBackend()
    messages: list[Message] = [
        SystemMessage(role="system", content="you are helpful"),
        UserMessage(role="user", content="hello"),
    ]
    monkeypatch.setattr(
        backend._transformers_backend,
        "generate",
        lambda **kwargs: AssistantMessage(role="assistant", content="ok"),
    )
    response = backend.generate(model="dummy", messages=messages)
    assert isinstance(response, AssistantMessage)
    assert response.content == "ok"
