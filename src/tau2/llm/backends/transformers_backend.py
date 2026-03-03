import json
import threading
import uuid
from typing import Any, Optional

from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.llm.backends.base import LLMBackend


class TransformersBackend(LLMBackend):
    name = "transformers"

    _cache_lock = threading.Lock()
    _inference_lock = threading.Lock()
    _model_cache: dict[str, tuple[Any, Any]] = {}

    def generate(
        self,
        model: str,
        messages: list[Message],
        tools: Optional[list[Tool]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> UserMessage | AssistantMessage:
        if tools and tool_choice is None:
            tool_choice = "auto"

        with self._inference_lock:
            text, usage, raw_data = self._generate_text(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )
        content, parsed_tool_calls = self._parse_assistant_output(
            text=text,
            tool_choice=tool_choice,
        )
        return AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=parsed_tool_calls,
            cost=0.0,
            usage=usage,
            raw_data=raw_data,
        )

    def _generate_text(
        self,
        model: str,
        messages: list[Message],
        tools: Optional[list[Tool]],
        tool_choice: Optional[str],
        **kwargs: Any,
    ) -> tuple[str, Optional[dict], dict]:
        tokenizer, hf_model = self._get_or_create_model_pair(
            model=model,
            kwargs=kwargs,
            auto_model_cls=AutoModelForCausalLM,
            auto_tokenizer_cls=AutoTokenizer,
        )

        chat_messages = self._to_chat_messages(messages)
        tool_schemas = [tool.openai_schema for tool in tools] if tools else None
        input_ids, attention_mask = self._build_model_inputs(
            tokenizer=tokenizer,
            chat_messages=chat_messages,
            tool_schemas=tool_schemas,
            tool_choice=tool_choice,
        )
        if attention_mask is None and hasattr(input_ids, "new_ones"):
            attention_mask = input_ids.new_ones(input_ids.shape)
        input_ids, attention_mask = self._move_to_model_device(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hf_model=hf_model,
        )

        generation_kwargs = self._build_generation_kwargs(
            tokenizer=tokenizer,
            kwargs=kwargs,
        )
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask
        output_ids = hf_model.generate(input_ids=input_ids, **generation_kwargs)
        prompt_tokens = self._sequence_length(input_ids)
        completion_ids = output_ids[:, prompt_tokens:]
        completion_tokens = self._sequence_length(completion_ids)
        text = tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        raw_data = {
            "backend": self.name,
            "model": model,
            "text": text,
            "tool_choice": tool_choice,
        }
        return text, usage, raw_data

    def _get_or_create_model_pair(
        self,
        model: str,
        kwargs: dict[str, Any],
        auto_model_cls: Any,
        auto_tokenizer_cls: Any,
    ) -> tuple[Any, Any]:
        model_load_kwargs = self._extract_model_load_kwargs(kwargs)
        cache_key = json.dumps(
            {"model": model, "kwargs": model_load_kwargs},
            sort_keys=True,
            default=str,
        )
        with self._cache_lock:
            cached = self._model_cache.get(cache_key)
            if cached is not None:
                return cached
            tokenizer = auto_tokenizer_cls.from_pretrained(
                model,
                trust_remote_code=model_load_kwargs.get("trust_remote_code", False),
                revision=model_load_kwargs.get("revision"),
            )
            hf_model = auto_model_cls.from_pretrained(model, **model_load_kwargs)
            pair = (tokenizer, hf_model)
            self._model_cache[cache_key] = pair
            return pair

    @staticmethod
    def _extract_model_load_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        keys = {
            "device_map",
            "torch_dtype",
            "trust_remote_code",
            "revision",
            "attn_implementation",
        }
        model_load_kwargs = {k: kwargs[k] for k in keys if k in kwargs}
        dtype = kwargs.get("dtype")
        if dtype is not None and "torch_dtype" not in model_load_kwargs:
            model_load_kwargs["torch_dtype"] = dtype
        if "device_map" not in model_load_kwargs:
            model_load_kwargs["device_map"] = "auto"
        return model_load_kwargs

    @classmethod
    def _build_model_inputs(
        cls,
        tokenizer: Any,
        chat_messages: list[dict[str, Any]],
        tool_schemas: Optional[list[dict[str, Any]]],
        tool_choice: Optional[str],
    ) -> tuple[Any, Optional[Any]]:
        if hasattr(tokenizer, "apply_chat_template"):
            rendered = None
            try:
                rendered = tokenizer.apply_chat_template(
                    chat_messages,
                    tools=tool_schemas,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            except Exception:
                rendered = None

            if rendered is not None:
                if hasattr(rendered, "input_ids"):
                    return rendered.input_ids, getattr(rendered, "attention_mask", None)
                if isinstance(rendered, dict) and "input_ids" in rendered:
                    return rendered["input_ids"], rendered.get("attention_mask")
                return rendered, None

        prompt = cls._build_fallback_prompt(chat_messages, tool_schemas, tool_choice)
        encoded = tokenizer(prompt, return_tensors="pt")
        if hasattr(encoded, "input_ids"):
            return encoded.input_ids, getattr(encoded, "attention_mask", None)
        if isinstance(encoded, dict) and "input_ids" in encoded:
            return encoded["input_ids"], encoded.get("attention_mask")
        raise ValueError("Tokenizer did not return input_ids")

    @classmethod
    def _move_to_model_device(
        cls, input_ids: Any, attention_mask: Optional[Any], hf_model: Any
    ) -> tuple[Any, Optional[Any]]:
        if not hasattr(input_ids, "to"):
            return input_ids, attention_mask
        device = getattr(hf_model, "device", None)
        if device is None:
            return input_ids, attention_mask
        moved_input_ids = input_ids.to(device)
        moved_attention_mask = (
            attention_mask.to(device)
            if attention_mask is not None and hasattr(attention_mask, "to")
            else attention_mask
        )
        return moved_input_ids, moved_attention_mask

    @staticmethod
    def _build_generation_kwargs(tokenizer: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        temperature = kwargs.get("temperature", 0.0)
        do_sample = kwargs.get("do_sample", temperature > 0.0)
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = kwargs.get("top_p", 1.0)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        elif eos_token_id is not None:
            generation_kwargs["pad_token_id"] = eos_token_id
        return generation_kwargs

    @staticmethod
    def _sequence_length(sequence: Any) -> int:
        if hasattr(sequence, "shape"):
            if len(sequence.shape) == 1:
                return int(sequence.shape[0])
            return int(sequence.shape[-1])
        if isinstance(sequence, list):
            if len(sequence) == 0:
                return 0
            if isinstance(sequence[0], list):
                return len(sequence[0])
            return len(sequence)
        return 0

    @staticmethod
    def _to_chat_messages(messages: list[Message]) -> list[dict[str, Any]]:
        chat_messages: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, UserMessage):
                chat_messages.append({"role": "user", "content": message.content or ""})
            elif isinstance(message, AssistantMessage):
                if message.tool_calls:
                    chat_messages.append(
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "tool_calls": [
                                        {
                                            "id": tc.id,
                                            "name": tc.name,
                                            "arguments": tc.arguments,
                                        }
                                        for tc in message.tool_calls
                                    ]
                                }
                            ),
                        }
                    )
                else:
                    chat_messages.append(
                        {"role": "assistant", "content": message.content or ""}
                    )
            elif isinstance(message, ToolMessage):
                chat_messages.append(
                    {
                        "role": "tool",
                        "content": message.content or "",
                        "tool_call_id": message.id,
                    }
                )
            elif isinstance(message, SystemMessage):
                chat_messages.append(
                    {"role": "system", "content": message.content or ""}
                )
        return chat_messages

    @staticmethod
    def _build_fallback_prompt(
        chat_messages: list[dict[str, Any]],
        tool_schemas: Optional[list[dict[str, Any]]],
        tool_choice: Optional[str],
    ) -> str:
        lines = []
        if tool_schemas:
            lines.append("Available tools:")
            lines.append(json.dumps(tool_schemas, indent=2))
            lines.append(
                "If you need a tool, output JSON: "
                '{"tool_calls":[{"name":"<tool_name>","arguments":{...}}]}'
            )
            if tool_choice == "required":
                lines.append("You must return a tool call.")
        for message in chat_messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines)

    def _parse_assistant_output(
        self,
        text: str,
        tool_choice: Optional[str],
    ) -> tuple[Optional[str], Optional[list[ToolCall]]]:
        stripped = text.strip()
        parsed_json = self._safe_load_json_block(stripped)
        tool_calls = self._extract_tool_calls(parsed_json)
        if tool_calls:
            content = None
            if isinstance(parsed_json, dict):
                raw_content = parsed_json.get("content")
                if isinstance(raw_content, str):
                    content = raw_content
            return content, tool_calls

        if tool_choice == "required":
            raise ValueError(
                "Model response did not contain a valid tool call while tool_choice='required'."
            )
        return stripped, None

    @staticmethod
    def _safe_load_json_block(text: str) -> Any:
        if text == "":
            return None
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None

    @staticmethod
    def _extract_tool_calls(parsed_json: Any) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        if isinstance(parsed_json, dict):
            if "tool_calls" in parsed_json and isinstance(parsed_json["tool_calls"], list):
                source_calls = parsed_json["tool_calls"]
            elif "name" in parsed_json and "arguments" in parsed_json:
                source_calls = [parsed_json]
            else:
                source_calls = []
        elif isinstance(parsed_json, list):
            source_calls = parsed_json
        else:
            source_calls = []

        for i, call in enumerate(source_calls):
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                arguments = function.get("arguments")
            else:
                name = call.get("name")
                arguments = call.get("arguments")
            if not isinstance(name, str):
                continue
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    logger.warning(
                        f"Failed to parse tool arguments string as JSON for tool {name}"
                    )
                    continue
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                continue
            tool_id = call.get("id") or f"call_{uuid.uuid4().hex}_{i}"
            tool_calls.append(
                ToolCall(
                    id=tool_id,
                    name=name,
                    arguments=arguments,
                )
            )
        return tool_calls
