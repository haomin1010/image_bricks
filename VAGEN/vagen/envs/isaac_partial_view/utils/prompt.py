"""
Prompt templates for the BrickIsaac partial-view environment.

The model can query one camera per turn, place one brick, or submit.
"""


def system_prompt(n_cameras: int = 5) -> str:
    """Return the system prompt describing the cube stacking task."""
    return f"""\
You are a robot arm controller. Your goal is to build a target block structure on a 6x6 tabletop grid.

At reset you first see all {n_cameras} target camera views (IDs 0..{n_cameras - 1}).
After reset, each query action can request exactly one camera view.
There are {n_cameras} cameras with IDs 0..{n_cameras - 1}.

Grid coordinates: x, y in {{0..5}}, z is the vertical layer (0 = bottom, 1 = one above, etc.).

Each turn output exactly ONE of:

1) Query one camera:
{{"query": [INT]}}

2) Place a cube:
{{"x": INT, "y": INT, "z": INT}}

3) When the structure is complete:
submit
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
[System]: Environment Reset. Study the target carefully.
Target multi-view images:
{camera_block}
Now query one camera per turn if needed, place the first cube, or submit if complete.
"""


def action_template(action_result: str, img_placeholder: str) -> str:
    """Template for the observation returned after a placement/parse branch."""
    return f"""\
[System]: {action_result}
Camera 0 view:
{img_placeholder}
You may query camera views, place the next cube, or submit.
"""


def query_result_template(camera_id: int, img_placeholder: str) -> str:
    """Template for observations returned after query actions."""
    return f"""\
[System]: Query result for camera {camera_id}.
{img_placeholder}
You may query one camera, place a cube, or submit.
"""


def format_prompt(n_cameras: int = 5, add_example: bool = True) -> str:
    """Generate output-format instructions appended to the system prompt."""
    base_prompt = f"""\
Each turn output exactly one action.
To query one camera: {{"query": [INT]}} (ID in 0..{n_cameras - 1})
To place a brick: {{"x": INT, "y": INT, "z": INT}}
When all bricks are placed correctly: submit
"""

    if add_example:
        examples = """
Examples:
  Query camera: {"query": [2]}
  Place a brick: {"x": 2, "y": 3, "z": 0}
  Submit: submit
"""
        return base_prompt + "\n" + examples

    return base_prompt


def _validate_system_prompt_text(text: str) -> bool:
    """Basic validation for both coordinate and query examples."""
    has_coord = '"x":' in text or '"x"' in text
    has_query = '"query"' in text
    return has_coord and has_query


def get_checked_system_prompt(n_cameras: int = 5, add_example: bool = True) -> str:
    """Return normal system prompt + format, otherwise a concise fallback."""
    base = system_prompt(n_cameras=n_cameras)
    fmt = format_prompt(n_cameras=n_cameras, add_example=add_example)
    composed = base + "\n" + fmt

    if _validate_system_prompt_text(composed):
        return composed

    corrective = (
        "System prompt validation failed. Please use one of the following formats:\n"
        f'Query one camera: {{"query": [0]}} (ID 0..{n_cameras - 1})\n'
        'Place a brick: {"x": INT, "y": INT, "z": INT}\n'
        "Submit: submit\n"
    )
    return corrective


def target_description(task_spec, max_attempts: int) -> str:
    """Build a concise target description from task metadata."""
    if int(getattr(task_spec, "total_blocks", 0)) <= 0:
        return (
            "Replicate the target block structure shown in the image. "
            "Place blocks one by one to match the target."
        )

    dims = tuple(getattr(task_spec, "dimensions", (0, 0, 0)))
    length = int(dims[0]) if len(dims) > 0 else 0
    width = int(dims[1]) if len(dims) > 1 else 0
    height = int(dims[2]) if len(dims) > 2 else 0
    total_blocks = int(getattr(task_spec, "total_blocks", 0))
    return (
        "Replicate the target structure shown in the images. "
        f"The target has {total_blocks} blocks in a {length}x{width}x{height} grid. "
        f"You may make at most {int(max_attempts)} placement attempts. "
        "Supported candidate placements are rewarded; floating or non-candidate placements are penalized."
    )
