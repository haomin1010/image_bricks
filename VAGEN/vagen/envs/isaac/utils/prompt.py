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


<<<<<<< HEAD
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
=======
def system_prompt(n_cameras: int = 3):
    """Return the system prompt describing the cube stacking task."""
    return f"""\
You are a robot arm controller. Your goal is to build a target block structure on a 6x6 tabletop grid.

At the start of each episode you are shown {n_cameras} camera views of the TARGET structure you must replicate.
You then place blocks one by one to recreate it.

Grid coordinates: x, y in {{0..5}}, z is the vertical layer (0 = bottom, 1 = one above, etc.).

Each turn output exactly ONE of:

1) Place a cube:
{{"x": INT, "y": INT, "z": INT}}

2) Query one or more cameras:
{{"query": [INT, ...]}}

3) When you believe the task is complete:
>>>>>>> main
submit
"""


<<<<<<< HEAD
def init_observation_template(img_placeholders: str):
    """Template for the initial observation shown after reset."""
    return f"""\
[System]: Environment Reset. All cubes are back to the pick position or hidden.
Current views:
{img_placeholders}
Please provide the coordinate (x,y,z) for the first cube (z=0). Examples (must match format exactly):
{{"x": INT, "y": INT, "z": INT}}
=======
def init_observation_template(img_placeholders: str, camera_labels: list = None):
    """Template for the initial observation shown after reset.

    Args:
        img_placeholders: One or more ``<image>`` placeholders, one per line.
        camera_labels: Optional list of camera label strings (e.g. ['top','front',...]).
    """
    if camera_labels:
        lines = []
        for label, ph in zip(camera_labels, img_placeholders.split("\n")):
            lines.append(f"{label}: {ph}")
        cam_section = "\n".join(lines)
    else:
        cam_section = img_placeholders
    return f"""\
[System]: Environment Reset. Study the TARGET structure carefully — these are the views you must replicate.
{cam_section}
Now place blocks one by one to reproduce the structure. Output {{"x": INT, "y": INT, "z": INT}} to place, {{"query": [INT, ...]}} to inspect cameras, or submit when done.
>>>>>>> main
"""



def action_template(action_result: str, img_placeholders: str):
    """Template for the observation returned after each step."""
    return f"""\
[System]: {action_result}
<<<<<<< HEAD
Updated views:
=======
Camera 0 view:
{img_placeholder}
You may query camera views, place the next cube, or submit."""


def query_result_template(camera_ids: list, img_placeholders: str):
    """Template for the observation returned after a query action.

    Args:
        camera_ids: List of camera IDs that were queried.
        img_placeholders: ``<image>`` placeholders (one per queried camera),
            separated by newlines.
    """
    cam_label = ", ".join(str(c) for c in camera_ids)
    return f"""\
[System]: Query result for camera(s) {cam_label}:
>>>>>>> main
{img_placeholders}
Determine the next coordinate. Examples (must match format exactly):
{{"x": INT, "y": INT, "z": INT}}"""


<<<<<<< HEAD

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
=======
def format_prompt(n_cameras: int = 3, add_example: bool = True):
    """Generate the output-format instructions appended to the system prompt."""
    base_prompt = """Each turn output exactly one action.
To place a brick: {"x": INT, "y": INT, "z": INT}
To inspect cameras: {"query": [INT, ...]}
When all bricks are placed correctly: submit"""

    if add_example:
        examples = """

Examples:
  Place a brick: {"x": 2, "y": 3, "z": 0}
  Query cameras: {"query": [0, 2]}
  Submit:        submit"""
        return base_prompt + examples
>>>>>>> main

    return base_prompt


def _validate_system_prompt_text(text: str) -> bool:
<<<<<<< HEAD
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
=======
    """Basic validation: prompt must contain a coordinate example."""
    return '"x":' in text or '"x"' in text
>>>>>>> main


def get_checked_system_prompt(add_example: bool = True) -> str:
    """Return the normal system prompt + format if valid, otherwise return a
<<<<<<< HEAD
    concise corrective example that shows the exact expected reply format.

    This helper is intended for Isaac-managed environments only: when the
    composed system prompt looks malformed, returning the short corrective
    example helps the agent produce a valid reply instead of entering a
    non-parseable dialogue loop.
=======
    concise corrective example.
>>>>>>> main
    """
    base = system_prompt()
    fmt = format_prompt(add_example=add_example)
    composed = base + "\n" + fmt

    if _validate_system_prompt_text(composed):
        return composed

    corrective = (
<<<<<<< HEAD
        "System prompt validation failed. Please use the following exact reply format:{{\"x\": INT, \"y\": INT, \"z\": INT}}\n\n"
        "Example:{{\"x\": 2, \"y\": 3, \"z\": 0}}\n"
=======
        'System prompt validation failed. Please use one of the following formats:\n'
        'Place a brick: {"x": INT, "y": INT, "z": INT}\n'
        'Submit: submit\n'
>>>>>>> main
    )

    return corrective