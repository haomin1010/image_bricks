"""
Parsing utilities for the BrickIsaac environment.

Extracts a single brick coordinate from an LLM response that follows
the ``<think>...</think><answer>{"x": INT, "y": INT, "z": INT}</answer>``
format.
"""

import json
import re
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Outer envelope: <think>...</think><answer>...</answer>
_ENVELOPE_RE = re.compile(
    r"<think>(.*?)</think>\s*<answer>(.*?)</answer>",
    re.DOTALL,
)

# Fallback: try to find a JSON object with x, y, z anywhere in the answer
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}")


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
    envelope_match = _ENVELOPE_RE.search(response)

    if not envelope_match:
        return {
            "llm_raw_response": response,
            "llm_response": response,
            "think_content": "",
            "action_content": "",
            "format_correct": False,
            "coordinate": None,
        }

    think_content = envelope_match.group(1).strip()
    action_content = envelope_match.group(2).strip()

    coordinate = _extract_coordinate(action_content)
    format_correct = coordinate is not None

    llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "format_correct": format_correct,
        "coordinate": coordinate,
    }


def _extract_coordinate(text: str) -> Optional[Dict[str, int]]:
    """Try to extract {"x": int, "y": int, "z": int} from *text*.

    Returns the coordinate dict, or ``None`` on failure.
    """
    # First try to parse the whole text as JSON
    coord = _try_parse_xyz(text)
    if coord is not None:
        return coord

    # Fallback: find any JSON object in the text
    for m in _JSON_OBJ_RE.finditer(text):
        coord = _try_parse_xyz(m.group())
        if coord is not None:
            return coord

    return None


def _try_parse_xyz(text: str) -> Optional[Dict[str, int]]:
    """Attempt to parse *text* as JSON and extract integer x, y, z."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(obj, dict):
        return None

    if "x" in obj and "y" in obj and "z" in obj:
        try:
            return {
                "x": int(obj["x"]),
                "y": int(obj["y"]),
                "z": int(obj["z"]),
            }
        except (ValueError, TypeError):
            return None

    return None

