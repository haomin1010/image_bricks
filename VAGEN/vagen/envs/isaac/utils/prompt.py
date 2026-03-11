"""
Prompt templates for the BrickIsaac environment.

The model places one brick per turn until it decides to submit.
After a place action the environment returns camera-0's view.
"""


def system_prompt(n_cameras: int = 3) -> str:
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


def init_observation_template(img_placeholders: str, camera_labels: list | None = None) -> str:
    """Template for the initial observation shown after reset."""
    if camera_labels:
        lines = []
        for label, ph in zip(camera_labels, img_placeholders.split("\n")):
            lines.append(f"{label}: {ph}")
        cam_section = "\n".join(lines)
    else:
        cam_section = img_placeholders
    return f"""\
[System]: Environment Reset. Study the TARGET structure carefully - these are the views you must replicate.
{cam_section}
Now place blocks one by one to reproduce the structure. Output {{"x": INT, "y": INT, "z": INT}} to place, or submit when done.
"""


def action_template(action_result: str, img_placeholder: str) -> str:
    """Template for the observation returned after a place action."""
    return f"""\
[System]: {action_result}
Camera 0 view:
{img_placeholder}
Place the next cube or submit when done."""


def query_result_template(camera_ids: list[int], img_placeholders: str) -> str:
    """Template for the observation returned after extra camera views."""
    cam_label = ", ".join(str(c) for c in camera_ids)
    return f"""\
[System]: Extra camera view(s) {cam_label}:
{img_placeholders}
Place the next cube or submit when done."""


def format_prompt(n_cameras: int = 3, add_example: bool = True) -> str:
    """Generate output-format instructions appended to the system prompt."""
    del n_cameras  # Keep signature for backward compatibility with callers.
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


def get_checked_system_prompt(n_cameras: int = 3, add_example: bool = True) -> str:
    """Return normal system prompt + format, otherwise a concise fallback."""
    base = system_prompt(n_cameras=n_cameras)
    fmt = format_prompt(n_cameras=n_cameras, add_example=add_example)
    composed = base + "\n" + fmt

    if _validate_system_prompt_text(composed):
        return composed

    corrective = (
        "System prompt validation failed. Please use one of the following formats:\n"
        'Place a brick: {"x": INT, "y": INT, "z": INT}\n'
        "Submit: submit\n"
    )
    return corrective


def target_description(task_spec, max_attempts: int) -> str:
    """Build a concise target description from task metadata."""
    if int(getattr(task_spec, "total_blocks", 0)) <= 0:
        return (
            "Your task is to replicate the block structure shown in the image. "
            "Observe the target configuration carefully and place blocks one by one "
            "to reproduce it."
        )

    dims = tuple(getattr(task_spec, "dimensions", (0, 0, 0)))
    length = int(dims[0]) if len(dims) > 0 else 0
    width = int(dims[1]) if len(dims) > 1 else 0
    height = int(dims[2]) if len(dims) > 2 else 0
    total_blocks = int(getattr(task_spec, "total_blocks", 0))
    return (
        "Your task is to replicate the target structure shown in the images. "
        f"The target contains {total_blocks} blocks in a {length}x{width}x{height} grid. "
        f"You may make at most {int(max_attempts)} placement attempts. "
        "A supported block on a valid target candidate is rewarded; floating or non-candidate placements are penalized."
    )
