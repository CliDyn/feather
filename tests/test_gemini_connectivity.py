"""Smoke test: verify that a Gemini API key can reach the Vertex AI Express
endpoint and return a coherent response.

Run with:
    pytest tests/test_gemini_connectivity.py -v -s --api-key AQ.xxx

Or set the env var and run without the flag:
    VERTEX_API_KEY=AQ.xxx pytest tests/test_gemini_connectivity.py -v -s

Marked as 'integration' so it is excluded from the default unit-test run.
"""

import os

import pytest


def _get_api_key(request):
    key = request.config.getoption("--api-key", default=None)
    if not key:
        key = os.getenv("VERTEX_API_KEY")
    return key


@pytest.mark.integration
def test_gemini_text_response(request):
    """Send a minimal text prompt; verify a non-empty string is returned."""
    api_key = _get_api_key(request)
    if not api_key:
        pytest.skip("No API key provided (--api-key or VERTEX_API_KEY)")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"

    response = client.models.generate_content(
        model=model,
        contents=["Reply with exactly three words: Gemini API works"],
        config=types.GenerateContentConfig(
            system_instruction="You are a concise assistant.",
        ),
    )

    text = response.text.strip()
    print(f"\nGemini response: {text!r}")

    assert text, "Empty response from Gemini"
    assert len(text) > 0
    print(f"[OK] Gemini API key is valid. Model: {model}")


@pytest.mark.integration
def test_gemini_json_parse(request):
    """Verify Gemini returns parseable JSON (same path as FigureAnalyzer)."""
    api_key = _get_api_key(request)
    if not api_key:
        pytest.skip("No API key provided (--api-key or VERTEX_API_KEY)")

    import json

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"

    response = client.models.generate_content(
        model=model,
        contents=['Return a JSON object with key "status" set to "ok".'],
        config=types.GenerateContentConfig(
            system_instruction="Return only valid JSON, no markdown fencing.",
        ),
    )

    text = response.text.strip()
    # Strip markdown fencing if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()

    data = json.loads(text)
    print(f"\nParsed JSON: {data}")
    assert data.get("status") == "ok", f"Unexpected JSON: {data}"
    print("[OK] JSON round-trip works.")
