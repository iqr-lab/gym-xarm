import mujoco
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
        self._success_thresh = 0.03
        self._goal = np.zeros(3, dtype=np.float32)
        super().__init__("reach", **kwargs)

    @property
    def goal(self):
        return self._goal

    def is_success(self):
        return np.linalg.norm(self.eef - self.goal) <= self._success_thresh

    def get_reward(self):
        dist = np.linalg.norm(self.eef - self.goal)
        return -dist

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

    def _set_goal_marker(self, goal_local):
        try:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "goal_marker",
            )
            if body_id == -1:
                print("Warning: goal_marker body not found")
                return

            goal_world = self.center_of_table + np.asarray(goal_local, dtype=np.float64)
            self.model.body_pos[body_id] = goal_world
            mujoco.mj_forward(self.model, self.data)
        except Exception as e:
            print(f"Warning: could not move goal marker: {e}")

    def _sample_goal(self):
        gripper_pos = np.array([1.280, 0.295, 0.735]) + self.np_random.uniform(
            -0.05, 0.05, size=3
        )
        super()._set_gripper(gripper_pos, self.gripper_rotation)

        # table-relative goal, matching self.eef frame
        goal = np.array([0.00, 0.00, 0.010], dtype=np.float32)
        goal[0] += self.np_random.uniform(-0.05, 0.05)
        goal[1] += self.np_random.uniform(-0.05, 0.05)
        goal[2] += self.np_random.uniform(-0.01, 0.10)

        self._goal = goal.astype(np.float32)
        self._set_goal_marker(self._goal)
        return self._goal

    def reset(self, seed=None, options: dict | None = None):
        self._action = np.zeros(4, dtype=np.float32)
        obs, info = super().reset(seed=seed, options=options)
        self._set_goal_marker(self._goal)
        return obs, info

    def step(self, action):
        self._action = action.copy()
        return super().step(action)