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
    """Return the system prompt describing the brick-building task."""
    return """\
You are a brick-building agent.

Task: You are given multiple camera views of a scene. Your goal is to place \
unit-cube bricks one at a time at integer grid coordinates (x, y, z) so that \
the final structure matches the target shape shown in the images.

Colour legend in the images:
  - Blue cells   → target positions that still need a brick
  - Green / Gold → bricks you have already placed correctly
  - Red cells    → bricks placed at wrong positions

Each turn you will receive observation images from multiple viewpoints \
(Top / Front / Side). Based on these images, decide where to place the next brick."""


def init_observation_template(img_placeholders: str):
    """Template for the initial observation shown after reset.

    Args:
        img_placeholders: A string containing n ``<image>`` placeholders
                          separated by newlines, one per camera view.
    """
    return f"""\
[Initial Observation]
Here are the current views of the scene:
{img_placeholders}
Decide where to place the next brick."""


def action_template(action_result: str, img_placeholders: str):
    """Template for the observation returned after each step.

    Args:
        action_result: A short description of what happened.
        img_placeholders: A string containing n ``<image>`` placeholders.
    """
    return f"""\
{action_result}
Updated views:
{img_placeholders}
Decide where to place the next brick."""


def format_prompt(add_example: bool = True):
    """Generate the output-format instructions appended to the system prompt."""
    base_prompt = """\
You must place exactly one brick per turn.
Your response should be in the format of:
<think>...</think><answer>{"x": INT, "y": INT, "z": INT}</answer>

Where x, y, z are the integer grid coordinates for the brick."""

    if add_example:
        examples = """
Example 1:
<think>Looking at the images, the base layer is missing a brick at position \
(2, 3, 0). I should place one there.</think>
<answer>{"x": 2, "y": 3, "z": 0}</answer>

Example 2:
<think>The first layer is complete. Now I need to start the second layer. \
Position (1, 1, 1) seems correct based on the reference.</think>
<answer>{"x": 1, "y": 1, "z": 1}</answer>
"""
        return base_prompt + "\n" + examples

    return base_prompt
