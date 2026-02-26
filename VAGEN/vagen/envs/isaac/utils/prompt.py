"""
Prompt templates for the BrickIsaac environment.

The LLM can perform two types of actions each turn:
  1. **Query** – request camera views by ID to inspect the scene.
  2. **Place** – output a coordinate to place the next brick.

After a *place* action the environment returns camera-0's view.
After a *query* action the environment returns the requested camera views.
"""


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

2) When you believe the task is complete:
submit
"""


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
Now place blocks one by one to reproduce the structure. Output {{"x": INT, "y": INT, "z": INT}} to place, or submit when done.
"""


def action_template(action_result: str, img_placeholder: str):
    """Template for the observation returned after a place action.

    Args:
        action_result: Textual feedback from the environment.
        img_placeholder: A single ``<image>`` placeholder for camera 0.
    """
    return f"""\
[System]: {action_result}
Camera 0 view:
{img_placeholder}
You may query camera views or place the next cube."""


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
{img_placeholders}
You may query more cameras, place a cube, or submit."""


def format_prompt(n_cameras: int = 3, add_example: bool = True):
    """Generate the output-format instructions appended to the system prompt."""
    base_prompt = """Each turn output exactly one action.
To place a brick: {"x": INT, "y": INT, "z": INT}
When all bricks are placed correctly: submit"""

    if add_example:
        examples = """

Examples:
  Place a brick: {"x": 2, "y": 3, "z": 0}
  Submit:        submit"""
        return base_prompt + examples

    return base_prompt


def _validate_system_prompt_text(text: str) -> bool:
    """Basic validation: prompt must contain a coordinate example."""
    return '"x":' in text or '"x"' in text


def get_checked_system_prompt(
    n_cameras: int = 3, add_example: bool = True
) -> str:
    """Return the normal system prompt + format if valid, otherwise return a
    concise corrective example.
    """
    base = system_prompt(n_cameras=n_cameras)
    fmt = format_prompt(n_cameras=n_cameras, add_example=add_example)
    composed = base + "\n" + fmt

    if _validate_system_prompt_text(composed):
        return composed

    corrective = (
        'System prompt validation failed. Please use one of the following formats:\n'
        'Place a brick: {"x": INT, "y": INT, "z": INT}\n'
        'Submit: submit\n'
    )
    return corrective
