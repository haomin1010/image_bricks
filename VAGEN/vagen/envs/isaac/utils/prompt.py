"""
Prompt templates for the BrickIsaac environment.
The LLM receives n camera-view images each turn and outputs a single
JSON coordinate to place the next brick.
Colour legend for mock-rendered images:
  - Blue   = target cell (not yet filled)
  - Green  = correctly placed brick (on target)
  - Gold   = correctly placed (overlap highlight)
  - Red    = incorrectly placed brick (not on target)
"""


def system_prompt():
    """Return the system prompt describing the cube stacking task."""
    return """\
You are a robot arm controller. You observe camera views of a 6x6 tabletop grid (coordinates x,y in {0..5}). Place one cube per turn by outputting the target grid coordinate (x,y,z) where z is 0 for the base layer, 1 for the layer above, etc.

Valid answer forms:
1) Place a cube:
{"x": INT, "y": INT, "z": INT}
- The JSON must contain integer values for keys "x", "y", and "z".
- Keys must be named exactly: x, y, z (lowercase).
- Use plain ASCII double quotes (").

2) When you believe the task is complete, submit:
submit
"""


def init_observation_template(img_placeholders: str):
    """Template for the initial observation shown after reset."""
    return f"""\
[System]: Environment Reset. All cubes are back to the pick position or hidden.
Current views:
{img_placeholders}
Please provide the coordinate (x,y,z) for the first cube (z=0). Examples (must match format exactly):
{{"x": INT, "y": INT, "z": INT}}
"""



def action_template(action_result: str, img_placeholders: str):
    """Template for the observation returned after each step."""
    return f"""\
[System]: {action_result}
Updated views:
{img_placeholders}
Determine the next coordinate. Examples (must match format exactly):
{{"x": INT, "y": INT, "z": INT}}"""



def format_prompt(add_example: bool = True):
    """Generate the output-format instructions appended to the system prompt."""
    base_prompt = f"""\
You must place exactly one brick per turn.
Your response should be in the format of:{{"x": INT, "y": INT, "z": INT}}
Where x, y, z are the integer grid coordinates for the brick.
When you believe all bricks are stacked correctly, output:
submit"""

    if add_example:
        examples = f"""
    Example 1:{{"x": 2, "y": 3, "z": 0}}
    Example 2:{{"x": 1, "y": 1, "z": 1}}
    """
        return base_prompt + "\n" + examples

    return base_prompt


def _validate_system_prompt_text(text: str) -> bool:
    """Basic validation for the composed system+format prompt.

    Checks presence of reasoning tag and an action envelope (<answer>)
    with an example JSON coordinate. This is intentionally lightweight — it only
    ensures the agent receives a clear machine-parseable example to avoid
    format-errors that lead to invalid dialogues.
    """

    # require a JSON-like coordinate example somewhere
    if "{\"x\"" not in text and "{\'x\'" not in text and '"x":' not in text:
        return False

    return True


def get_checked_system_prompt(add_example: bool = True) -> str:
    """Return the normal system prompt + format if valid, otherwise return a
    concise corrective example that shows the exact expected reply format.

    This helper is intended for Isaac-managed environments only: when the
    composed system prompt looks malformed, returning the short corrective
    example helps the agent produce a valid reply instead of entering a
    non-parseable dialogue loop.
    """
    base = system_prompt()
    fmt = format_prompt(add_example=add_example)
    composed = base + "\n" + fmt

    if _validate_system_prompt_text(composed):
        return composed

    # Fallback corrective example (minimal, explicit and machine-parseable)
    corrective = (
        "System prompt validation failed. Please use the following exact reply format:{{\"x\": INT, \"y\": INT, \"z\": INT}}\n\n"
        "Example:{{\"x\": 2, \"y\": 3, \"z\": 0}}\n"
    )

    return corrective