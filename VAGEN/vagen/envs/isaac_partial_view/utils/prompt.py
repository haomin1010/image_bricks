"""
Prompt templates for the BrickIsaac partial-view environment.

The model can query one camera per turn, place one brick, or submit.
"""


def system_prompt(n_cameras: int = 5) -> str:
    """Return the system prompt describing the cube stacking task."""
    return f"""\
You are a robot arm controller. Your goal is to build a target block structure on a 8x8 tabletop grid.

At reset you first see all {n_cameras} target camera views (IDs 0..{n_cameras - 1}).
Grid coordinates: x, y in {{0..7}}, z is the vertical layer (0 = bottom, 1 = one above, etc.).
"""


def init_observation_template(img_placeholders: str, camera_labels: list | None = None) -> str:
    """Template for the initial observation shown after reset."""
    if camera_labels:
        lines = []
        for label, ph in zip(camera_labels, img_placeholders.split("\n")):
            lines.append(f"{label}: {ph}")
        camera_block = "\n".join(lines)
    else:
        camera_block = img_placeholders

    return f"""\
Target multi-view images:
{camera_block}
From the current state, you must query at least one camera before each placement or submit action.
Query additional views if needed, then choose the next action that best advances the build toward the target.
"""


def action_template(action_result: str, img_placeholder: str) -> str:
    """Template for the observation returned after a placement/parse branch."""
    return f"""\
{action_result}
You must query at least one camera before your next placement or submit action.
You may query one camera, place a cube, or submit.
"""


def query_result_template(camera_id: int, img_placeholder: str) -> str:
    """Template for observations returned after query actions."""
    return f"""\
Query result for camera {camera_id}.
{img_placeholder}
You have queried a camera for this turn. Query another camera if needed, or place a cube / submit when ready.
"""


def format_prompt(n_cameras: int = 5, add_example: bool = True) -> str:
    """Generate output-format instructions appended to the system prompt."""
    base_prompt = f"""\
Each turn output exactly one action in this format:
<thinking></thinking><action>...</action>

Use the thinking section to briefly reason about the target views, the current partial structure, and the next best action before acting.
Think step by step and keep the thinking concise and directly relevant to the next action.

Valid action content inside <action> is exactly ONE of:

1) Query one camera:
{{"query": [INT]}}

2) Place a cube:
{{"x": INT, "y": INT, "z": INT}}

3) When the structure is complete:
submit
"""

    if add_example:
        examples = """
Examples:
  Query camera: <thinking></thinking><action>{"query": [2]}</action>
  Place a brick: <thinking></thinking><action>{"x": 2, "y": 3, "z": 0}</action>
  Submit: <thinking></thinking><action>submit</action>
"""
        return base_prompt + "\n" + examples

    return base_prompt


def _validate_system_prompt_text(text: str) -> bool:
    """Basic validation for both coordinate and query examples."""
    has_coord = '"x":' in text or '"x"' in text
    has_query = '"query"' in text
    has_tags = "<thinking>" in text and "<action>" in text
    return has_coord and has_query and has_tags


def get_checked_system_prompt(n_cameras: int = 5, add_example: bool = True) -> str:
    """Return normal system prompt + format, otherwise a concise fallback."""
    base = system_prompt(n_cameras=n_cameras)
    fmt = format_prompt(n_cameras=n_cameras, add_example=add_example)
    composed = base + "\n" + fmt

    if _validate_system_prompt_text(composed):
        return composed

    corrective = (
        "System prompt validation failed. Please use one of the following formats:\n"
        f'Query one camera: <thinking></thinking><action>{{"query": [0]}}</action> (ID 0..{n_cameras - 1})\n'
        '<thinking></thinking><action>{"x": INT, "y": INT, "z": INT}</action>\n'
        "<thinking></thinking><action>submit</action>\n"
    )
    return corrective


def target_description(task_spec, max_attempts: int) -> str:
    """Build a concise target description from task metadata."""
    return (
        "Build the target block structure shown in the reference images. "
        "Use the target views to infer the correct shape and continue building from the current state."
    )
