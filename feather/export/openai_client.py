"""Thin OpenAI API wrapper with retry logic.

Follows the same pattern as the Gemini wrapper in ``llm/analyzer.py``
but targets the OpenAI chat completions API.
"""

import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY_S = 10


class OpenAIClient:
    """Wrapper around the OpenAI chat completions API.

    Parameters
    ----------
    config : dict
        Report config dict with ``model``, ``max_tokens``, ``temperature`` keys.
    api_key : str or None
        Explicit API key. Falls back to ``OPENAI_API_KEY`` env var
        (or the env var named by ``config["api_key_env"]``).
    """

    def __init__(self, config: dict, api_key: str | None = None) -> None:
        from openai import OpenAI

        self.config = config
        key = api_key or os.getenv(
            config.get("api_key_env", "OPENAI_API_KEY"), ""
        )
        if not key:
            raise RuntimeError(
                "OpenAI API key not found. Set the 'OPENAI_API_KEY' "
                "environment variable or pass api_key= to OpenAIClient."
            )
        self.client = OpenAI(api_key=key)

    def chat_json(self, system: str, user: str) -> dict:
        """Send a chat completion request and parse the JSON response.

        Parameters
        ----------
        system : str
            System prompt.
        user : str
            User prompt.

        Returns
        -------
        dict
            Parsed JSON from the LLM response.
        """
        text = self._call(system=system, user=user)
        return self._parse_json(text)

    def _call(self, system: str, user: str) -> str:
        """Call OpenAI with retry on transient errors."""
        model = self.config.get("model", "gpt-4o")
        max_tokens = self.config.get("max_tokens", 16384)
        temperature = self.config.get("temperature", 0.3)

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return response.choices[0].message.content
            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "OpenAI call failed (attempt %d/%d): %s — "
                        "retrying in %ds",
                        attempt, _MAX_RETRIES, exc, _RETRY_DELAY_S,
                    )
                    time.sleep(_RETRY_DELAY_S)
                else:
                    raise
        raise RuntimeError("OpenAI call failed after all retries")

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse JSON from LLM response, stripping markdown fencing and
        fixing invalid escape sequences."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fix invalid escape sequences (same approach as analyzer.py)
        cleaned = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', cleaned)
        cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned)

        return json.loads(cleaned)
