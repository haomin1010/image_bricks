"""
Parsing utilities for the BrickIsaac environment.
Extracts a single brick coordinate from an LLM response that follows
the ``<think>...</think><answer>{"x": INT, "y": INT, "z": INT}</answer>``
format.
"""

import json
import re
from typing import Dict, Optional


# Strict-only mode: no fallback JSON scanning.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_response(response: str) -> Dict:
    """Parse an LLM response and extract a brick coordinate.
    Args:
        response: Raw LLM output string.
    Returns:
        A dict with the following keys:
        - ``llm_raw_response`` (str): The original response.
        - ``llm_response`` (str): Reconstructed canonical response.
        - ``think_content`` (str): Content inside ``<think>`` tags.
        - ``action_content`` (str): Content inside ``<answer>`` tags.
        - ``format_correct`` (bool): Whether the envelope + JSON were valid.
        - ``coordinate`` (dict | None): ``{"x": int, "y": int, "z": int}``
          or ``None`` if parsing failed.
    """
    print("Parsing response:", response)

    action_content = response

    coordinate = _extract_coordinate(action_content)
    is_submit = _is_submit(action_content)
    format_correct = coordinate is not None or is_submit

    print("Extracted coordinate:", coordinate)
    print("Is submit action:", is_submit)
    print("Format correct:", format_correct)

    return {
        "llm_raw_response": response,
        "action_content": action_content,
        "format_correct": format_correct,
        "coordinate": coordinate,
        "is_submit": is_submit,
    }


def _is_submit(text: str) -> bool:
    """Check if the action content indicates submission."""
    clean = text.lower().strip()
    if clean == "submit":
        return True
    
    # Check for {"action": "submit"}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and obj.get("action") == "submit":
            return True
    except (json.JSONDecodeError, TypeError):
        pass
        
    return False


def _extract_coordinate(text: str) -> Optional[Dict[str, int]]:
    """Parse a strict {"x": int, "y": int, "z": int} JSON object.

    Only the exact JSON object is accepted. Any extra text or keys causes
    parsing to fail.
    """
    return _try_parse_xyz(text)


def _try_parse_xyz(text: str) -> Optional[Dict[str, int]]:
    """Attempt to parse *text* as JSON and extract integer x, y, z."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(obj, dict):
        return None

    if set(obj.keys()) == {"x", "y", "z"}:
        try:
            return {
                "x": int(obj["x"]),
                "y": int(obj["y"]),
                "z": int(obj["z"]),
            }
        except (ValueError, TypeError):
            return None

    return None