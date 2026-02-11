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
You are a robot arm controller.
Task: You see multiple camera views of a tabletop with a grey grid and some blue cubes.
Your goal is to stack all the blue cubes on top of each other at specific grid coordinates (x, y). 
Grid: The grid is 6x6, with coordinates from (0,0) to (5,5).
Position (3,3) is the center of the grid.
Each turn, you must decide the (x, y, z) coordinate where the next cube should be placed.
z=0 is the bottom-most position. z=1 is on top of the first cube, and so on.
IMPORTANT: Output only the required JSON format and thinking process. Do not include any vision tags or special image tokens in your response.
"""


def init_observation_template(img_placeholders: str):
    """Template for the initial observation shown after reset."""
    return f"""\
[System]: Environment Reset. All cubes are back to the pick position or hidden.
Current views:
{img_placeholders}
Please provide the coordinate (x,y,z) for the first cube (z=0)."""



def action_template(action_result: str, img_placeholders: str):
    """Template for the observation returned after each step."""
    return f"""\
[System]: {action_result}
Updated views:
{img_placeholders}
Determine the next (x, y, z) coordinate."""



def format_prompt(add_example: bool = True):
    """Generate the output-format instructions appended to the system prompt."""
    base_prompt = """\
You must place exactly one brick per turn.
Your response should be in the format of:
<think>...</think><answer>{"x": INT, "y": INT, "z": INT}</answer>
Where x, y, z are the integer grid coordinates for the brick.
When you believe all bricks are stacked correctly, output:
<think>...</think><answer>submit</answer>"""

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