"""
Parsing utilities for the BrickIsaac environment.
Extracts an action from an LLM response.  Supported action types:

1. **Place brick**  – ``<thinking>...</thinking><annotation>...</annotation><action>{"x": INT, "y": INT, "z": INT}</action>``
2. **Query camera** – ``<thinking>...</thinking><annotation>...</annotation><action>{"query": [INT]}</action>``
3. **Submit**      – ``<thinking>...</thinking><annotation>...</annotation><action>submit</action>``
"""

import json
import re
from typing import Dict, List, Optional, Tuple


# Strict-only mode: no fallback JSON scanning.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_response(response: str) -> Dict:
    """Parse an LLM response and extract an action.

    Args:
        response: Raw LLM output string.

    Returns:
        A dict with the following keys:

        - ``llm_raw_response`` (str): The original response.
        - ``action_content`` (str): The raw action text.
        - ``format_correct`` (bool): Whether a valid action was found.
        - ``coordinate`` (dict | None): ``{"x": int, "y": int, "z": int}``
          or ``None``.
        - ``query_cameras`` (list[int] | None): Camera IDs to query, or
          ``None`` if this is not a query action.
        - ``is_submit`` (bool): Whether the action is a submit.
    """
    print("Parsing response:", response)

    thinking_content, annotation_content, action_content = _extract_tagged_sections(response)
    has_required_tags = (
        thinking_content is not None
        and annotation_content is not None
        and action_content is not None
    )

    coordinate = _extract_coordinate(action_content) if has_required_tags else None
    query_cameras = _extract_query(action_content) if has_required_tags else None
    is_submit = _is_submit(action_content) if has_required_tags else False
    format_correct = (
        has_required_tags
        and (coordinate is not None or query_cameras is not None or is_submit)
    )

    print("Extracted thinking content:", thinking_content)
    print("Extracted annotation content:", annotation_content)
    print("Extracted coordinate:", coordinate)
    print("Extracted query cameras:", query_cameras)
    print("Is submit action:", is_submit)
    print("Format correct:", format_correct)

    return {
        "llm_raw_response": response,
        "action_content": action_content,
        "thinking_content": thinking_content,
        "annotation_content": annotation_content,
        "format_correct": format_correct,
        "coordinate": coordinate,
        "query_cameras": query_cameras,
        "is_submit": is_submit,
    }


def _extract_tagged_sections(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract ``<thinking>``, ``<annotation>``, and ``<action>`` contents from a strict response."""
    if not isinstance(text, str):
        return None, None, None

    match = re.fullmatch(
        r"\s*<thinking>(?P<thinking>.*?)</thinking>\s*<annotation>(?P<annotation>.*?)</annotation>\s*<action>(?P<action>.*?)</action>\s*",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return None, None, None

    return (
        match.group("thinking").strip(),
        match.group("annotation").strip(),
        match.group("action").strip(),
    )


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


def _extract_query(text: str) -> Optional[List[int]]:
    """Parse a strict ``{"query": [INT]}`` JSON object.

    Returns a single-item camera-ID list if valid, otherwise ``None``.
    The list must contain exactly one non-negative integer.
    """
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(obj, dict):
        return None

    if set(obj.keys()) != {"query"}:
        return None

    cam_list = obj["query"]
    if not isinstance(cam_list, list) or len(cam_list) != 1:
        return None

    try:
        cam_id = int(cam_list[0])
    except (ValueError, TypeError):
        return None

    if cam_id < 0:
        return None

    return [cam_id]
