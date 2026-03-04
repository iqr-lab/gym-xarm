import numpy as np

from gym_xarm.tasks import Base


class Reach(Base):
    metadata = {
        **Base.metadata,
        "action_space": "xyzw",
        "episode_length": 50,
        "description": "Move the end-effector to a target position",
    }

    def __init__(self, **kwargs):
        # success threshold in meters
        self._success_thresh = 0.03
        self._goal = np.zeros(3, dtype=np.float32)
        super().__init__("reach", **kwargs)

    @property
    def goal(self):
        return self._goal

    def is_success(self):
        return np.linalg.norm(self.eef - self.goal) <= self._success_thresh

    def get_reward(self):
        # dense shaping reward: negative distance
        dist = np.linalg.norm(self.eef - self.goal)

        # small bonus when close + encourage opening gripper slightly (optional)
        close_bonus = 1.0 if dist <= self._success_thresh else 0.0
        gripper_bonus = max(self._action[-1], 0.0) / 50.0  # same scale idea as Lift

        # scale similar to Lift: distance is ~0.0-0.5, divide to keep magnitudes tame
        return (-dist) / 10.0 + close_bonus + gripper_bonus

    def _get_obs(self):
        eef_to_goal = self.eef - self.goal
        dist = np.linalg.norm(eef_to_goal)
        dist_xy = np.linalg.norm(eef_to_goal[:-1])

        return np.concatenate(
            [
                self.eef,
                self.eef_velp,
                self.goal,
                eef_to_goal,
                np.array([dist, dist_xy, self._success_thresh], dtype=np.float32),
                self.gripper_angle,
            ],
            axis=0,
        )

    def _sample_goal(self):
        # -------------------------
        # Randomize gripper start
        # -------------------------
        gripper_pos = np.array([1.280, 0.295, 0.735]) + self.np_random.uniform(
            -0.05, 0.05, size=3
        )
        super()._set_gripper(gripper_pos, self.gripper_rotation)

        # -------------------------
        # Sample reachable goal
        # -------------------------
        # Goal near the table center, slightly above table surface
        # You may need to tweak these offsets based on your xArm setup.
        goal = self.center_of_table.copy()
        goal += np.array([0.05, 0.00, 0.10])  # bias forward + above table
        goal[0] += self.np_random.uniform(-0.10, 0.10)
        goal[1] += self.np_random.uniform(-0.10, 0.10)
        goal[2] += self.np_random.uniform(-0.03, 0.08)

        self._goal = goal.astype(np.float32)

        # Base expects a "goal" return (even if you don't use it elsewhere)
        return self._goal

    def reset(self, seed=None, options: dict | None = None):
        self._action = np.zeros(4, dtype=np.float32)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self._action = action.copy()
        return super().step(action)