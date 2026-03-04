import numpy as np

from gym_xarm.tasks import Base


class _PickPlaceBase(Base):
    metadata = {
        **Base.metadata,
        "action_space": "xyzw",
        "episode_length": 75,
        "description": "Pick a cube and place it at a target position",
    }

    def __init__(self, reward_mode: str, **kwargs):
        assert reward_mode in ("dense", "semi", "sparse")
        self._reward_mode = reward_mode

        # thresholds (tune as needed)
        self._lift_z_thresh = 0.10        # lift above table by this much
        self._place_tol = 0.04            # xy+z tolerance for success
        self._goal = np.zeros(3, dtype=np.float32)
        self._init_z = 0.0

        super().__init__("pick_place", **kwargs)

    @property
    def goal(self):
        return self._goal

    @property
    def lift_z_target(self):
        return self._init_z + self._lift_z_thresh

    def is_success(self):
        # success: object close to goal
        return np.linalg.norm(self.obj - self.goal) <= self._place_tol

    def get_reward(self):
        # distances
        reach_dist = np.linalg.norm(self.obj - self.eef)
        place_dist = np.linalg.norm(self.obj - self.goal)

        lifted = self.obj[2] >= (self.lift_z_target - 0.005)
        obj_dropped = (self.obj[2] < (self._init_z + 0.005)) and (reach_dist > 0.03)

        # -------------------------
        # Sparse reward
        # -------------------------
        if self._reward_mode == "sparse":
            return 1.0 if self.is_success() else 0.0

        # -------------------------
        # Semi-sparse staged reward
        # -------------------------
        if self._reward_mode == "semi":
            r = 0.0

            # encourage reaching early
            r += (-reach_dist) / 10.0

            # stage bonuses
            if lifted and not obj_dropped:
                r += 1.0
            if place_dist < self._place_tol and not obj_dropped:
                r += 2.0  # placing is the main goal

            return r

        # -------------------------
        # Dense shaping reward
        # -------------------------
        # Idea:
        # - stage 1: reach (minimize eef->obj)
        # - stage 2: lift (maximize z up to target)
        # - stage 3: place (minimize obj->goal)
        # Keep magnitudes stable like Lift: distances divided down.
        r = 0.0

        # Reach shaping
        r += (-reach_dist) / 10.0

        # Lift shaping
        if not obj_dropped:
            lift_progress = np.clip((self.obj[2] - self._init_z) / max(self._lift_z_thresh, 1e-6), 0.0, 1.0)
            r += 0.5 * lift_progress

        # Place shaping (only meaningful once lifted a bit)
        if self.obj[2] > (self._init_z + 0.02) and not obj_dropped:
            r += (-place_dist) / 10.0

        # Completion bonus
        if self.is_success() and not obj_dropped:
            r += 3.0

        # tiny gripper term (optional)
        r += max(self._action[-1], 0.0) / 100.0

        return r

    def _get_obs(self):
        eef_to_obj = self.eef - self.obj
        obj_to_goal = self.obj - self.goal

        return np.concatenate(
            [
                self.eef,
                self.eef_velp,
                self.obj,
                self.obj_rot,
                self.obj_velp,
                self.obj_velr,
                self.goal,
                eef_to_obj,
                obj_to_goal,
                np.array(
                    [
                        np.linalg.norm(eef_to_obj),
                        np.linalg.norm(eef_to_obj[:-1]),
                        np.linalg.norm(obj_to_goal),
                        np.linalg.norm(obj_to_goal[:-1]),
                        self.lift_z_target,
                        self.lift_z_target - self.obj[-1],
                    ],
                    dtype=np.float32,
                ),
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
        # Randomize object start
        # -------------------------
        object_pos = self.center_of_table - np.array([0.15, 0.10, 0.07])
        object_pos[0] += self.np_random.uniform(-0.06, 0.06)
        object_pos[1] += self.np_random.uniform(-0.06, 0.06)

        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object_joint0")
        object_qpos[:3] = object_pos
        self._utils.set_joint_qpos(self.model, self.data, "object_joint0", object_qpos)

        self._init_z = float(object_pos[2])

        # -------------------------
        # Randomize goal on table (same z as object start)
        # -------------------------
        goal = self.center_of_table.copy()
        goal += np.array([0.05, 0.00, -0.07])  # move into reachable tabletop region
        goal[0] += self.np_random.uniform(-0.10, 0.10)
        goal[1] += self.np_random.uniform(-0.10, 0.10)
        goal[2] = object_pos[2]  # goal on table plane

        self._goal = goal.astype(np.float32)

        return self._goal

    def reset(self, seed=None, options: dict | None = None):
        self._action = np.zeros(4, dtype=np.float32)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self._action = action.copy()
        return super().step(action)


class PickPlaceDense(_PickPlaceBase):
    metadata = {**_PickPlaceBase.metadata, "description": "Pick and place with dense shaping reward"}

    def __init__(self, **kwargs):
        super().__init__(reward_mode="dense", **kwargs)


class PickPlaceSemi(_PickPlaceBase):
    metadata = {**_PickPlaceBase.metadata, "description": "Pick and place with semi-sparse staged reward"}

    def __init__(self, **kwargs):
        super().__init__(reward_mode="semi", **kwargs)


class PickPlaceSparse(_PickPlaceBase):
    metadata = {**_PickPlaceBase.metadata, "description": "Pick and place with sparse success-only reward"}

    def __init__(self, **kwargs):
        super().__init__(reward_mode="sparse", **kwargs)