"""
Ouroboros — LLM client.

The only module that communicates with external LLM APIs.
Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-3-pro-preview"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Fetch current pricing from OpenRouter API.

    Returns dict of {model_id: (input_per_1m, cached_per_1m, output_per_1m)}.
    Returns empty dict on failure.
    """
    import logging
    log = logging.getLogger("ouroboros.llm")

    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}

    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        models = data.get("data", [])

        # Prefixes we care about
        prefixes = ("anthropic/", "openai/", "google/", "meta-llama/", "x-ai/", "qwen/")

        pricing_dict = {}
        for model in models:
            model_id = model.get("id", "")
            if not model_id.startswith(prefixes):
                continue

            pricing = model.get("pricing", {})
            if not pricing or not pricing.get("prompt"):
                continue

            # OpenRouter pricing is in dollars per token (raw values)
            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str else None

            # Convert to per-million tokens
            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            if raw_cached is not None:
                cached_price = round(raw_cached * 1_000_000, 4)
            else:
                cached_price = round(prompt_price * 0.1, 4)  # fallback: 10% of prompt

            # Sanity check: skip obviously wrong prices
            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            pricing_dict[model_id] = (prompt_price, cached_price, completion_price)

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


class LLMClient:
    """LLM API wrapper. Routes Gemini models to Google; others to OpenRouter."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL,
    ):
        self._openrouter_api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._openrouter_base_url = base_url
        self._openrouter_client = None
        self._gemini_client = None

    @staticmethod
    def _is_gemini_model(model: str) -> bool:
        return "gemini" in str(model or "").strip().lower()

    @staticmethod
    def _normalize_gemini_model_name(model: str) -> str:
        """
        Strip provider prefixes before calling Google's OpenAI-compatible endpoint.

        Examples:
        - gemini/gemini-3.1-pro-preview -> gemini-3.1-pro-preview
        - google/gemini-2.5-pro -> gemini-2.5-pro
        """
        name = str(model or "").strip()
        if not name:
            return name

        # Remove common provider wrappers first.
        while "/" in name:
            head, tail = name.split("/", 1)
            if head.lower() in {"google", "gemini", "models", "openrouter"}:
                name = tail
                continue
            break

        # If any segment still contains a Gemini model ID, keep that segment.
        if "/" in name:
            for seg in name.split("/"):
                if seg.lower().startswith("gemini"):
                    return seg
        return name

    @staticmethod
    def _infer_schema_type_from_value(value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "string"

    @classmethod
    def _sanitize_json_schema_for_gemini(cls, schema: Any) -> None:
        """Mutate JSON schema in-place to satisfy Gemini's strict validator."""
        if not isinstance(schema, dict):
            return

        if schema.get("required") == []:
            schema.pop("required", None)

        properties = schema.get("properties")
        if isinstance(properties, dict):
            if "type" not in schema:
                schema["type"] = "object"
            for key, prop in list(properties.items()):
                if not isinstance(prop, dict):
                    properties[key] = {"type": "string"}
                    continue
                if "type" not in prop:
                    if isinstance(prop.get("properties"), dict):
                        prop["type"] = "object"
                    elif "items" in prop:
                        prop["type"] = "array"
                    elif isinstance(prop.get("default"), (bool, int, float, str, list, dict)):
                        prop["type"] = cls._infer_schema_type_from_value(prop["default"])
                    elif isinstance(prop.get("enum"), list) and prop["enum"]:
                        prop["type"] = cls._infer_schema_type_from_value(prop["enum"][0])
                    else:
                        prop["type"] = "string"
                cls._sanitize_json_schema_for_gemini(prop)

        if "items" in schema:
            if "type" not in schema:
                schema["type"] = "array"
            items = schema["items"]
            if isinstance(items, dict):
                if "type" not in items and isinstance(items.get("properties"), dict):
                    items["type"] = "object"
                cls._sanitize_json_schema_for_gemini(items)
            elif isinstance(items, list):
                for item in items:
                    cls._sanitize_json_schema_for_gemini(item)

        for key in ("anyOf", "allOf", "oneOf"):
            variants = schema.get(key)
            if isinstance(variants, list):
                for variant in variants:
                    cls._sanitize_json_schema_for_gemini(variant)

        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            cls._sanitize_json_schema_for_gemini(additional)

    @classmethod
    def _sanitize_tools_for_gemini(cls, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized = copy.deepcopy(tools)
        for tool in sanitized:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function")
            if isinstance(fn, dict):
                params = fn.get("parameters")
                if isinstance(params, dict):
                    cls._sanitize_json_schema_for_gemini(params)
        return sanitized

    @staticmethod
    def _messages_for_gemini(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenRouter-friendly message payload into Gemini-compatible format.

        - Flattens multipart system/developer content to plain text.
        - Removes cache_control keys from content blocks.
        """
        normalized: List[Dict[str, Any]] = []

        for original in messages:
            if not isinstance(original, dict):
                continue
            msg = copy.deepcopy(original)
            role = str(msg.get("role") or "").strip().lower()
            content = msg.get("content")

            if isinstance(content, list):
                cleaned_blocks = []
                for block in content:
                    if isinstance(block, dict):
                        block.pop("cache_control", None)
                    cleaned_blocks.append(block)

                if role in {"system", "developer"}:
                    parts: List[str] = []
                    for block in cleaned_blocks:
                        if isinstance(block, dict):
                            if block.get("type") == "text" and "text" in block:
                                parts.append(str(block.get("text") or ""))
                            elif "text" in block:
                                parts.append(str(block.get("text") or ""))
                        elif isinstance(block, str):
                            parts.append(block)
                    msg["content"] = "\n\n".join([p for p in parts if p]).strip()
                else:
                    msg["content"] = cleaned_blocks
            elif role in {"system", "developer"} and not isinstance(content, str):
                msg["content"] = "" if content is None else str(content)

            normalized.append(msg)

        return normalized

    def _get_openrouter_client(self):
        if self._openrouter_client is None:
            from openai import OpenAI
            self._openrouter_client = OpenAI(
                base_url=self._openrouter_base_url,
                api_key=self._openrouter_api_key,
                default_headers={
                    "HTTP-Referer": "https://colab.research.google.com/",
                    "X-Title": "Ouroboros",
                },
            )
        return self._openrouter_client

    def _get_gemini_client(self):
        if self._gemini_client is None:
            from openai import OpenAI
            gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not gemini_key:
                raise RuntimeError("GEMINI_API_KEY is not set for Gemini model call")
            self._gemini_client = OpenAI(
                base_url=GEMINI_OPENAI_BASE_URL,
                api_key=gemini_key,
            )
        return self._gemini_client

    def _get_client_and_model(self, model: str) -> Tuple[Any, str, bool]:
        if self._is_gemini_model(model):
            return self._get_gemini_client(), self._normalize_gemini_model_name(model), True
        return self._get_openrouter_client(), model, False

    def _fetch_generation_cost(self, generation_id: str) -> Optional[float]:
        """Fetch cost from OpenRouter Generation API as fallback."""
        try:
            import requests
            url = f"{self._openrouter_base_url.rstrip('/')}/generation?id={generation_id}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._openrouter_api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            # Generation might not be ready yet — retry once after short delay
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._openrouter_api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)
            pass
        return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call. Returns: (response_message_dict, usage_dict with cost)."""
        client, api_model, is_gemini = self._get_client_and_model(model)
        effort = normalize_reasoning_effort(reasoning_effort)

        kwargs: Dict[str, Any] = {
            "model": api_model,
            "messages": self._messages_for_gemini(messages) if is_gemini else messages,
            "max_tokens": max_tokens,
        }

        if not is_gemini:
            extra_body: Dict[str, Any] = {
                "reasoning": {"effort": effort, "exclude": True},
            }
            # Pin Anthropic models to Anthropic provider for prompt caching
            if model.startswith("anthropic/"):
                extra_body["provider"] = {
                    "order": ["Anthropic"],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                }
            kwargs["extra_body"] = extra_body

        if tools:
            if is_gemini:
                tools_with_cache = self._sanitize_tools_for_gemini(tools)
            else:
                # Add cache_control to last tool for Anthropic prompt caching.
                tools_with_cache = [t for t in tools]  # shallow copy
                if tools_with_cache:
                    last_tool = {**tools_with_cache[-1]}  # copy last tool
                    last_tool["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                    tools_with_cache[-1] = last_tool
            kwargs["tools"] = tools_with_cache
            kwargs["tool_choice"] = tool_choice

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        # Extract cached_tokens from prompt_tokens_details if available
        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])

        # Extract cache_write_tokens from prompt_tokens_details if available
        # OpenRouter: "cache_write_tokens"
        # Native Anthropic: "cache_creation_tokens" or "cache_creation_input_tokens"
        if not usage.get("cache_write_tokens"):
            prompt_details_for_write = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_for_write, dict):
                cache_write = (prompt_details_for_write.get("cache_write_tokens")
                              or prompt_details_for_write.get("cache_creation_tokens")
                              or prompt_details_for_write.get("cache_creation_input_tokens"))
                if cache_write:
                    usage["cache_write_tokens"] = int(cache_write)

        # Ensure cost is present in usage for OpenRouter calls.
        if (not is_gemini) and (not usage.get("cost")):
            gen_id = resp_dict.get("id") or ""
            if gen_id:
                cost = self._fetch_generation_cost(gen_id)
                if cost is not None:
                    usage["cost"] = cost

        return msg, usage

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "anthropic/claude-sonnet-4.6",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Send a vision query to an LLM. Lightweight — no tools, no loop.

        Args:
            prompt: Text instruction for the model
            images: List of image dicts. Each dict must have either:
                - {"url": "https://..."} — for URL images
                - {"base64": "<b64>", "mime": "image/png"} — for base64 images
            model: VLM-capable model ID
            max_tokens: Max response tokens
            reasoning_effort: Effort level

        Returns:
            (text_response, usage_dict)
        """
        # Build multipart content
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"]},
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                })
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))

        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        return os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models
