from gymnasium.envs.registration import register

# -----------------------------
# Reach
# -----------------------------
register(
    id="gym_xarm/XarmReach-v0",
    entry_point="gym_xarm.tasks:Reach",
    max_episode_steps=300,
    kwargs={"obs_type": "state"},
)

# -----------------------------
# Pick & Place (3 reward variants)
# -----------------------------
register(
    id="gym_xarm/XarmPickPlaceDense-v0",
    entry_point="gym_xarm.tasks:PickPlaceDense",
    max_episode_steps=300,
    kwargs={"obs_type": "state"},
)

register(
    id="gym_xarm/XarmPickPlaceSemi-v0",
    entry_point="gym_xarm.tasks:PickPlaceSemi",
    max_episode_steps=300,
    kwargs={"obs_type": "state"},
)

register(
    id="gym_xarm/XarmPickPlaceSparse-v0",
    entry_point="gym_xarm.tasks:PickPlaceSparse",
    max_episode_steps=300,
    kwargs={"obs_type": "state"},
)

# -----------------------------
# Existing Lift task
# -----------------------------
register(
    id="gym_xarm/XarmLift-v0",
    entry_point="gym_xarm.tasks:Lift",
    max_episode_steps=300,
    kwargs={"obs_type": "state"},
)