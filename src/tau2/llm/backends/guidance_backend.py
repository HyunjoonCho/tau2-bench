from typing import Any, Optional

from loguru import logger

from tau2.data_model.message import AssistantMessage, Message, UserMessage
from tau2.environment.tool import Tool
from tau2.llm.backends.base import LLMBackend
from tau2.llm.backends.transformers_backend import TransformersBackend


class GuidanceBackend(LLMBackend):
    """
    Guidance backend scaffold.

    This backend currently validates Guidance availability and delegates generation to
    the transformers backend. Guidance-specific constraints can be layered on top in
    a follow-up iteration.
    """

    name = "guidance"

    def __init__(self):
        self._transformers_backend = TransformersBackend()

    def generate(
        self,
        model: str,
        messages: list[Message],
        tools: Optional[list[Tool]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> UserMessage | AssistantMessage:
        self._ensure_guidance_installed()
        logger.warning(
            "Guidance backend is running with baseline transformers behavior. "
            "Constraint-specific guidance programs can be added in a follow-up pass."
        )
        return self._transformers_backend.generate(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    @staticmethod
    def _ensure_guidance_installed() -> None:
        try:
            import guidance  # noqa: F401
        except Exception as e:
            raise ImportError(
                "Guidance backend requires the `guidance` package. "
                "Install it before using --*-llm-backend guidance."
            ) from e

