"""OpenAI-style schemas for the intuitive robot tools served by server.py."""

_OBSERVATION_ID = {"type": "string", "description": "The newest observation_id returned by any robot tool."}
_COMMAND_ID = {"type": "string", "description": "Unique id; reusing it with identical arguments returns its receipt."}
_VECTOR3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "observe",
            "description": "Return labelled RGB camera images, the measured TCP pose and gripper state, the task, "
            "and the robot card (frames, limits, cameras).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_to",
            "description": "Move the TCP by a RELATIVE base-frame displacement [m] and a RELATIVE base-frame "
            "axis-angle rotation [rad] about the current TCP, anchored to the measured pose. Blocks until the arm "
            "settles; the receipt reports completed, partial or rejected and the actual residual.",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_id": _OBSERVATION_ID,
                    "command_id": _COMMAND_ID,
                    "translation_m": {**_VECTOR3, "description": "Base-frame [dx, dy, dz] in metres."},
                    "rotation_delta_rad": {**_VECTOR3, "description": "Base-frame axis-angle vector in radians."},
                    "frame": {"type": "string", "enum": ["base"]},
                    "note": {"type": "string", "description": "Optional free-text intent; not executed."},
                },
                "required": ["observation_id", "command_id", "translation_m"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gripper",
            "description": "Open or close the gripper. A closed gripper is not proof of a grasp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_id": _OBSERVATION_ID,
                    "command_id": _COMMAND_ID,
                    "action": {"type": "string", "enum": ["open", "close"]},
                },
                "required": ["observation_id", "command_id", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "batch_actions",
            "description": "Run at most three literal move_to/gripper actions in order, then observe. Only observe "
            "may follow a gripper action. Stops at the first action that does not complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_id": _OBSERVATION_ID,
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": {"type": "string", "enum": ["move_to", "gripper", "observe"]},
                                "arguments": {"type": "object"},
                            },
                            "required": ["tool"],
                        },
                    },
                },
                "required": ["observation_id", "actions"],
            },
        },
    },
]
