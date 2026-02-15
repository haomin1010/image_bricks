"""
Prompt templates for the BrickIsaac environment.

The LLM can perform two types of actions each turn:
  1. **Query** – request camera views by ID to inspect the scene.
  2. **Place** – output a coordinate to place the next brick.

After a *place* action the environment returns camera-0's view.
After a *query* action the environment returns the requested camera views.
"""


def system_prompt(n_cameras: int = 3):
    """Return the system prompt describing the cube stacking task.

    Args:
        n_cameras: Total number of available cameras (IDs 0 .. n_cameras-1).
    """
    return f"""\
You are a robot arm controller. You observe camera views of a 6x6 tabletop grid (coordinates x,y in {{0..5}}). Your goal is to build a target shape by placing one cube at a time.

You have {n_cameras} cameras available (IDs 0 to {n_cameras - 1}).

Each turn you must output exactly ONE of the following actions:

1) Query camera views:
{{"query": [CAM_ID, ...]}}
- CAM_ID must be integers in {{0..{n_cameras - 1}}}.
- You may request one or more cameras in a single query.
- The environment will return the requested camera images.

2) Place a cube:
{{"x": INT, "y": INT, "z": INT}}
- x, y are grid coordinates in {{0..5}}.
- z is the layer (0 = base, 1 = above, etc.).
- The environment will return camera 0's view after placement.

3) When you believe the task is complete:
submit
"""


def init_observation_template(img_placeholder: str):
    """Template for the initial observation shown after reset.

    Args:
        img_placeholder: A single ``<image>`` placeholder for camera 0.
    """
    return f"""\
[System]: Environment Reset. All cubes are back to the pick position or hidden.
Camera 0 view:
{img_placeholder}
You may query other camera views or place the first cube.
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
    """Generate the output-format instructions appended to the system prompt.

    Args:
        n_cameras: Total number of available cameras.
        add_example: Whether to append concrete examples.
    """
    base_prompt = f"""\
Each turn you must output exactly one action.
To query cameras: {{"query": [CAM_ID, ...]}}  (IDs in 0..{n_cameras - 1})
To place a brick: {{"x": INT, "y": INT, "z": INT}}
When all bricks are placed correctly: submit"""

    if add_example:
        examples = f"""

Examples:
  Query cameras 0 and 2: {{"query": [0, 2]}}
  Place a brick:         {{"x": 2, "y": 3, "z": 0}}
  Submit:                submit"""
        return base_prompt + examples

    return base_prompt


def _validate_system_prompt_text(text: str) -> bool:
    """Basic validation for the composed system+format prompt.

    Checks that the prompt contains both a coordinate example and a query
    example so the agent knows both action formats.
    """
    has_coord = '"x":' in text or '"x"' in text
    has_query = '"query"' in text
    return has_coord and has_query


def get_checked_system_prompt(
    n_cameras: int = 3, add_example: bool = True
) -> str:
    """Return the normal system prompt + format if valid, otherwise return a
    concise corrective example that shows the exact expected reply format.
    """
    base = system_prompt(n_cameras=n_cameras)
    fmt = format_prompt(n_cameras=n_cameras, add_example=add_example)
    composed = base + "\n" + fmt

    if _validate_system_prompt_text(composed):
        return composed

    # Fallback corrective example (minimal, explicit and machine-parseable)
    corrective = (
        'System prompt validation failed. Please use one of the following formats:\n'
        f'Query cameras: {{"query": [0, 1]}}  (IDs 0..{n_cameras - 1})\n'
        'Place a brick: {"x": INT, "y": INT, "z": INT}\n'
        'Submit: submit\n'
    )
    return corrective
