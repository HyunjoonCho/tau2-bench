from abc import ABC, abstractmethod
from typing import Any, Optional

from tau2.data_model.message import AssistantMessage, Message, UserMessage
from tau2.environment.tool import Tool


class LLMBackend(ABC):
    """Interface for pluggable LLM backends."""

    name: str

    @abstractmethod
    def generate(
        self,
        model: str,
        messages: list[Message],
        tools: Optional[list[Tool]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> UserMessage | AssistantMessage:
        """Generate a response in Tau2 message format."""
        raise NotImplementedError

