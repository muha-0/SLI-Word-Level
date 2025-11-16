"""
Grammar / text correction module using Sapling API.

Set the environment variable SAPLING_API_KEY before using:
    export SAPLING_API_KEY="your-key-here"
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

import requests

SAPLING_API_URL = "https://api.sapling.ai/api/v1/edits"
SAPLING_API_KEY = os.getenv("SAPLING_API_KEY")


def _apply_corrections(text: str, corrections: List[Dict[str, Any]]) -> str:
    """Apply Sapling edits to the original text."""
    offset = 0
    chars = list(text)

    for edit in corrections:
        start = edit["start"] + offset
        end = edit["end"] + offset
        replacement = edit["replacement"]

        # Replace the slice in a way that preserves indices for subsequent edits
        chars[start:end] = list(replacement)
        offset += len(replacement) - (end - start)

    return "".join(chars)


def correct_text(text: str) -> str:
    """
    Send text to Sapling for grammar/style correction.

    If SAPLING_API_KEY is not set or the API fails, returns the original text.
    """
    if not SAPLING_API_KEY:
        # Fail gracefully if no key is configured
        return text

    payload = {
        "key": SAPLING_API_KEY,
        "text": text,
        "session_id": "sli-esl-demo",
    }

    try:
        resp = requests.post(SAPLING_API_URL, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        corrections = data.get("edits", [])
        return _apply_corrections(text, corrections)
    except Exception as e:
        # In production you might log this. For now we simply return the original.
        print(f"[Sapling] Error during correction: {e}")
        return text


if __name__ == "__main__":
    sample = "perfecto"
    print("Original :", sample)
    print("Corrected:", correct_text(sample))