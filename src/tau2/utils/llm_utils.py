from typing import Any, Optional

from loguru import logger

from tau2.data_model.message import AssistantMessage, Message, ToolMessage, UserMessage
from tau2.environment.tool import Tool
from tau2.llm.backends import GuidanceBackend, LLMBackend, LiteLLMBackend, TransformersBackend
from tau2.llm.backends.litellm_backend import to_litellm_messages, to_tau2_messages

_BACKENDS: dict[str, LLMBackend] = {
    "litellm": LiteLLMBackend(),
    "transformers": TransformersBackend(),
    "guidance": GuidanceBackend(),
}


def register_backend(name: str, backend: LLMBackend) -> None:
    """Register or replace an LLM backend implementation."""
    _BACKENDS[name] = backend


def get_backend(name: str) -> LLMBackend:
    """Get a registered LLM backend by name."""
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown LLM backend: {name}. Available backends: {list(_BACKENDS.keys())}"
        )
    return _BACKENDS[name]


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    llm_backend: str = "litellm",
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the selected LLM backend.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        llm_backend: Backend name to use (litellm, transformers, guidance).
        **kwargs: Additional backend-specific arguments.
    """
    backend = get_backend(llm_backend)
    return backend.generate(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        **kwargs,
    )


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0.0
    user_cost = 0.0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage

