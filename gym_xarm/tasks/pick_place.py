import numpy as np
import mujoco
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
        self._place_tol = 0.015            # xy+z tolerance for success
        self._goal = np.zeros(3, dtype=np.float32)
        self._init_z = 0.0

        super().__init__("pick_place", **kwargs)

    @property
    def goal(self):
        return self._goal - self.center_of_table

    @property
    def lift_z_target(self):
        return self._init_z + self._lift_z_thresh
    

    def _set_goal_marker(self, goal_local):

        site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "goal_site"
        )

        if site_id < 0:
            return

        table_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "table0"
        )
        table_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "table0"
        )

        table_pos = self.model.body_pos[table_body_id]

        goal_body = goal_local - table_pos
        goal_body[2] = goal_body[2] -0.0215

        # print(f"table pos {table_pos}")
        # print(f"goal body {goal_body}")

        self.model.site_pos[site_id] = goal_body

        mujoco.mj_forward(self.model, self.data)

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
        # Pick & Place Dense Reward
        # -------------------------

        # distances
        reach_dist = np.linalg.norm(self.eef - self.obj)
        place_dist = np.linalg.norm(self.obj - self.goal)

        # height info
        obj_height = self.obj[2]
        table_height = self._init_z

        # lifted flag (state-based, NOT action-based)
        lifted = obj_height > table_height + 0.04

        # -------------------------
        # 1. Reach reward (always active)
        # -------------------------
        r_reach = -reach_dist

        # -------------------------
        # 2. Lift reward (only when close)
        # -------------------------
        # encourage lifting ONLY when near object
        if reach_dist < 0.05:
            lift_height = np.clip(obj_height - table_height, 0.0, 0.2)
            r_lift = lift_height
        else:
            r_lift = 0.0

        # -------------------------
        # 3. Place reward (only when lifted)
        # -------------------------
        if lifted:
            r_place = -place_dist
        else:
            r_place = 0.0

        # -------------------------
        # 4. Combine (balanced weights)
        # -------------------------
        r = (
            1.0 * r_reach
            + 2.0 * r_lift
            + 5.0 * r_place
        )

        # -------------------------
        # 5. Success bonus (dominant)
        # -------------------------
        if self.is_success():
            r += 10.0

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
        object_pos[0] += self.np_random.uniform(-0.1, 0.1)
        object_pos[1] += self.np_random.uniform(-0.1, 0.1)

        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object_joint0")
        object_qpos[:3] = object_pos
        self._utils.set_joint_qpos(self.model, self.data, "object_joint0", object_qpos)

        self._init_z = float(object_pos[2])

        # -------------------------
        # Randomize goal on table (same z as object start)
        # -------------------------
        goal = self.center_of_table - np.array([0.15, -0.10, 0.07], dtype=np.float32)
        goal[0] += self.np_random.uniform(-0.10, 0.10)
        goal[1] += self.np_random.uniform(-0.30, 0.10)

        self._goal = goal.astype(np.float32)
        self._set_goal_marker(self._goal)
        return self._goal

    def reset(self, seed=None, options: dict | None = None):
        self._action = np.zeros(4, dtype=np.float32)
        self._prev_eef = self.eef.copy()
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