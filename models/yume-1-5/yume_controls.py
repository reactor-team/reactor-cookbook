"""Native prompt encoding, independent of GPU and runtime dependencies."""

from yume_model import Movement, View

MOVEMENT_TEXT: dict[Movement, str] = {
    "none": "The camera's movement direction remains stationary (·).",
    "forward": "The camera pushes forward (W).",
    "backward": "The camera pulls back (S).",
    "left": "The camera moves to the left (A).",
    "right": "The camera moves to the right (D).",
    "forward_left": "The camera pushes forward and moves to the left (W+A).",
    "forward_right": "The camera pushes forward and moves to the right (W+D).",
    "backward_left": "The camera pulls back and moves to the left (S+A).",
    "backward_right": "The camera pulls back and moves to the right (S+D).",
}
VIEW_TEXT: dict[View, str] = {
    "none": "The rotation direction of the camera remains stationary (·).",
    "pan_left": "The camera pans to the left (←).",
    "pan_right": "The camera pans to the right (→).",
    "tilt_up": "The camera tilts up (↑).",
    "tilt_down": "The camera tilts down (↓).",
    "tilt_up_left": "The camera tilts up and pans to the left (↑←).",
    "tilt_up_right": "The camera tilts up and pans to the right (↑→).",
    "tilt_down_left": "The camera tilts down and pans to the left (↓←).",
    "tilt_down_right": "The camera tilts down and pans to the right (↓→).",
}


def conditioned_prompt(prompt: str, movement: Movement, view: View) -> str:
    """Encode controls in the caption format used to train and sample YUME."""
    distance = 0 if movement == "none" else 4
    rotation = 0 if view == "none" else 4
    return " ".join(
        (
            "First-person perspective.",
            MOVEMENT_TEXT[movement],
            VIEW_TEXT[view],
            f"Actual distance moved:{distance} at 100 meters per second.",
            f"Angular change rate (turn speed):{rotation}.",
            f"View rotation speed:{rotation}.",
            prompt.strip(),
        )
    )
