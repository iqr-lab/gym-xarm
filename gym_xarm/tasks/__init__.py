from collections import OrderedDict

from gym_xarm.tasks.base import Base as Base
from gym_xarm.tasks.lift import Lift
from gym_xarm.tasks.reach import Reach
from gym_xarm.tasks.pick_place import PickPlaceDense, PickPlaceSemi, PickPlaceSparse

TASKS = OrderedDict(
    (
        (
            "reach",
            {
                "env": Reach,
                "action_space": "xyzw",
                "episode_length": 50,
                "description": "Reach a target location with the end effector",
            },
        ),
        (
            "pick_place_dense",
            {
                "env": PickPlaceDense,
                "action_space": "xyzw",
                "episode_length": 75,
                "description": "Pick and place with dense shaping reward",
            },
        ),
        (
            "pick_place_semi",
            {
                "env": PickPlaceSemi,
                "action_space": "xyzw",
                "episode_length": 75,
                "description": "Pick and place with semi-sparse staged reward",
            },
        ),
        (
            "pick_place_sparse",
            {
                "env": PickPlaceSparse,
                "action_space": "xyzw",
                "episode_length": 75,
                "description": "Pick and place with sparse success-only reward",
            },
        ),
        (
            "lift",
            {
                "env": Lift,
                "action_space": "xyzw",
                "episode_length": 50,
                "description": "Lift a cube above a height threshold",
            },
        ),
    )
)