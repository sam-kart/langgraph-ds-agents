import time
from datetime import datetime, timezone

import anthropic

_client = None
MODEL = "claude-sonnet-4-6"

_usage = {
    "calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}
_calls = []


def _add_usage(response) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    _usage["calls"] += 1
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        _usage[key] += getattr(usage, key, 0) or 0


def get_usage_summary() -> dict:
    """Return cumulative Anthropic token usage for the current process."""
    return {**_usage, "calls_detail": list(_calls)}


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def complete(system: str, user: str, max_tokens: int = 4096) -> str:
    """Call Claude with the system prompt marked for caching.

    Anthropic caches any content block tagged cache_control=ephemeral whose
    token count meets the minimum (1024 tokens for Sonnet). Subsequent calls
    with the same system prompt hit the cache at ~10% of normal input cost.
    """
    started = time.perf_counter()
    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user}],
    )
    duration = round(time.perf_counter() - started, 3)
    _add_usage(response)
    usage = getattr(response, "usage", None)
    _calls.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "duration_seconds": duration,
            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
            "cache_creation_input_tokens": (
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            ),
            "cache_read_input_tokens": (
                getattr(usage, "cache_read_input_tokens", 0) or 0
            ),
        }
    )
    return response.content[0].text
