from __future__ import annotations
import sys
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Any
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# ====================== 【新增】屏蔽警告 ======================
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====================== 【新增】SB3 PPO 相关 ======================
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
import os

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orca_sim.versions import (
    resolve_scene_path,
)

# ====================== 【不变】基类环境 ======================
class BaseOrcaHandEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        scene_file: str,
        version: str | None = None,
        frame_skip: int = 5,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.scene_path = resolve_scene_path(scene_file, version=version)
        self.version = self.scene_path.parent.name
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)

        self._default_camera = "closeup"
        self._renderer: mujoco.Renderer | None = None
        self._viewer: Any | None = None

        ctrl_range = self.model.actuator_ctrlrange.copy()
        self.action_low = ctrl_range[:, 0].astype(np.float32)
        self.action_high = ctrl_range[:, 1].astype(np.float32)
        self.action_space = spaces.Box(
            low=self.action_low,
            high=self.action_high,
            dtype=np.float32,
        )

        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float64,
        )

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def _get_reward(self) -> float:
        return 0.0

    def _get_terminated(self) -> bool:
        return False

    def _get_truncated(self) -> bool:
        return False

    def _get_info(self) -> dict[str, Any]:
        return {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if options and "qpos" in options:
            qpos = np.asarray(options["qpos"], dtype=np.float64)
            if qpos.shape != self.data.qpos.shape:
                raise ValueError(
                    f"Expected qpos shape {self.data.qpos.shape}, got {qpos.shape}"
                )
            self.data.qpos[:] = qpos

        if options and "qvel" in options:
            qvel = np.asarray(options["qvel"], dtype=np.float64)
            if qvel.shape != self.data.qvel.shape:
                raise ValueError(
                    f"Expected qvel shape {self.data.qvel.shape}, got {qvel.shape}"
                )
            self.data.qvel[:] = qvel

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        self.data.ctrl[:] = np.clip(action, self.action_low, self.action_high)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer
                self._viewer = viewer.launch_passive(self.model, self.data)
                mujoco.mjv_defaultFreeCamera(self.model, self._viewer.cam)
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

# ====================== 【不变】任务环境 ======================
class OrcaHandRightCubeOrientation(BaseOrcaHandEnv):
    DEFAULT_INITIAL_RED_FACE = "down"
    DEFAULT_CUBE_POS_XY_JITTER = np.array([0.0, 0.0], dtype=np.float64)
    RED_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    RED_FACE_LOCAL_NORMAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def __init__(
        self,
        render_mode: str | None = None,
        version: str | None = None,
        *,
        scene_file: str = "scene_right_cube_orientation.xml",
        cube_joint_name: str = "cube_freejoint",
        cube_body_name: str = "task_cube",
        hand_pose_by_joint: Mapping[str, float] | None = None,
        initial_red_face: str = DEFAULT_INITIAL_RED_FACE,
        cube_pos_xy_jitter: float | tuple[float, float] = 0.0,
        max_episode_steps: int = 500,
        success_tolerance_rad: float = np.deg2rad(15.0),
        drop_height: float = 0.05,
    ) -> None:
        self.scene_file = scene_file
        self.cube_joint_name = cube_joint_name
        self.cube_body_name = cube_body_name
        self._requested_hand_pose_by_joint = None if hand_pose_by_joint is None else dict(hand_pose_by_joint)
        self.initial_red_face = self._validate_initial_red_face(initial_red_face)
        self.cube_pos_xy_jitter = self._normalize_xy_jitter(cube_pos_xy_jitter)
        self.max_episode_steps = max_episode_steps
        self.success_tolerance_rad = float(success_tolerance_rad)
        self.drop_height = float(drop_height)
        self._elapsed_steps = 0
        super().__init__(scene_file, version=version, frame_skip=5, render_mode=render_mode)

        self._cube_joint_id = self.model.joint(self.cube_joint_name).id
        self._cube_qpos_adr = int(self.model.jnt_qposadr[self._cube_joint_id])
        self._cube_qvel_adr = int(self.model.jnt_dofadr[self._cube_joint_id])
        self._cube_body_id = self.model.body(self.cube_body_name).id
        self._actuator_qpos_indices = self._resolve_actuator_qpos_indices()
        self._default_cube_pos = self.model.qpos0[self._cube_qpos_adr : self._cube_qpos_adr+3].copy()
        self._default_cube_quat = self._normalize_quat(self.model.qpos0[self._cube_qpos_adr+3 : self._cube_qpos_adr+7].copy())

        if self._requested_hand_pose_by_joint is None:
            self._default_hand_qpos = self.model.qpos0[: self._cube_qpos_adr].copy()
            self._hand_pose_by_joint = self._extract_hand_pose_by_joint(self._default_hand_qpos)
        else:
            self._hand_pose_by_joint = dict(self._requested_hand_pose_by_joint)
            self._default_hand_qpos = self._build_hand_qpos(self._hand_pose_by_joint)

        obs = self._get_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float64)

    def _resolve_actuator_qpos_indices(self) -> np.ndarray:
        indices = np.empty(self.model.nu, dtype=np.int32)
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("Only hinge joints supported")
            indices[actuator_id] = int(self.model.jnt_qposadr[joint_id])
        return indices

    def _build_hand_qpos(self, pose_by_joint: Mapping[str, float]) -> np.ndarray:
        hand_qpos = self.model.qpos0[: self._cube_qpos_adr].copy()
        for joint_name, joint_value in pose_by_joint.items():
            joint_id = self.model.joint(joint_name).id
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            hand_qpos[qpos_adr] = float(joint_value)
        return hand_qpos

    def _extract_hand_pose_by_joint(self, hand_qpos: np.ndarray) -> dict[str, float]:
        pose_by_joint = {}
        for actuator_id, qpos_adr in enumerate(self._actuator_qpos_indices):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            pose_by_joint[self.model.joint(joint_id).name] = float(hand_qpos[qpos_adr])
        return pose_by_joint

    def _resolve_default_cube_pos(self, jitter_xy: np.ndarray) -> np.ndarray:
        cube_pos = self._default_cube_pos.copy()
        if np.any(jitter_xy):
            cube_pos[:2] += self.np_random.uniform(low=-jitter_xy, high=jitter_xy)
        return cube_pos

    def nominal_reset_options(self) -> dict[str, Any]:
        return {
            "hand_pose_by_joint": dict(self._hand_pose_by_joint),
            "cube_pos": self._default_cube_pos.copy(),
            "cube_quat": self._default_cube_quat.copy(),
            "settle_steps": 0,
        }

    def sample_randomized_reset_options(self, seed=None, initial_red_face="random", cube_pos_xy_jitter=None):
        rng = np.random.default_rng(seed)
        jitter_xy = self.cube_pos_xy_jitter.copy() if cube_pos_xy_jitter is None else self._normalize_xy_jitter(cube_pos_xy_jitter)
        cube_pos = self._default_cube_pos.copy()
        if np.any(jitter_xy):
            cube_pos[:2] += rng.uniform(low=-jitter_xy, high=jitter_xy)
        initial_red_face = self._validate_initial_red_face(initial_red_face)
        cube_quat = self._default_cube_quat.copy() if initial_red_face == "down" else self._sample_random_nonsolved_quaternion(rng)
        return {"hand_pose_by_joint": dict(self._hand_pose_by_joint), "cube_pos": cube_pos, "cube_quat": cube_quat, "settle_steps": 0}

    def _resolve_initial_cube_quat(self, options):
        if "cube_quat" in options:
            return self._normalize_quat(np.asarray(options["cube_quat"], dtype=np.float64))
        initial_red_face = self._validate_initial_red_face(options.get("initial_red_face", self.initial_red_face))
        return self._default_cube_quat.copy() if initial_red_face == "down" else self._sample_random_nonsolved_quaternion(self.np_random)

    def _compose_ctrl_from_qpos(self):
        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        for actuator_id, qpos_idx in enumerate(self._actuator_qpos_indices):
            ctrl[actuator_id] = float(np.clip(self.data.qpos[qpos_idx], self.action_low[actuator_id], self.action_high[actuator_id]))
        return ctrl

    def _cube_quat(self): return self.data.qpos[self._cube_qpos_adr+3 : self._cube_qpos_adr+7].copy()
    def _cube_pos(self): return self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr+3].copy()
    def _cube_qvel(self): return self.data.qvel[self._cube_qvel_adr : self._cube_qvel_adr+6].copy()

    def _cube_red_face_world_normal(self):
        quat = self._normalize_quat(self._cube_quat())
        w,x,y,z = quat
        return np.array([2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)], dtype=np.float64)

    def _red_face_up_alignment(self): return float(np.dot(self._cube_red_face_world_normal(), self.WORLD_UP))
    def _red_face_up_angle_rad(self): return float(np.arccos(np.clip(self._red_face_up_alignment(), -1,1)))
    def _goal_reached(self): return bool(self._red_face_up_alignment() >= np.cos(self.success_tolerance_rad))
    def _cube_dropped(self): return bool(self.data.xpos[self._cube_body_id,2] < self.drop_height)

    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()
        if not hasattr(self, "_cube_qpos_adr"): return base
        return np.concatenate([base, self._cube_red_face_world_normal(), np.array([self._red_face_up_alignment()], dtype=np.float64)])

    def _get_reward(self) -> float:
        align = 0.5 * (self._red_face_up_alignment() + 1)
        lift = np.clip(self.data.xpos[self._cube_body_id,2]-0.12, 0,0.12)/0.12
        drop = 1.0 if self._cube_dropped() else 0.0
        return float(align + 0.1*lift - drop)

    def _get_terminated(self): return self._goal_reached() or self._cube_dropped()
    def _get_truncated(self): return self._elapsed_steps >= self.max_episode_steps

    def _get_info(self):
        return {
            #"is_success": self._goal_reached(),
            "dropped": self._cube_dropped(),
            "red_face_up_alignment": self._red_face_up_alignment(),
            "elapsed_steps": self._elapsed_steps
        }

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._elapsed_steps = 0
        options = options or {}
        full_qpos, full_qvel = options.get("qpos"), options.get("qvel")

        if full_qpos is not None:
            self.data.qpos[:] = np.asarray(full_qpos, dtype=np.float64)
        else:
            hand_qpos = self._default_hand_qpos.copy()
            if "hand_pose_by_joint" in options: hand_qpos = self._build_hand_qpos(options["hand_pose_by_joint"])
            if "hand_qpos" in options: hand_qpos = np.asarray(options["hand_qpos"], dtype=np.float64)
            cube_pos = np.asarray(options["cube_pos"], dtype=np.float64) if "cube_pos" in options else self._resolve_default_cube_pos(self.cube_pos_xy_jitter)
            cube_quat = self._resolve_initial_cube_quat(options)
            self.data.qpos[:self._cube_qpos_adr] = hand_qpos
            self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr+3] = cube_pos
            self.data.qpos[self._cube_qpos_adr+3 : self._cube_qpos_adr+7] = cube_quat

        self.data.qvel[:] = np.asarray(full_qvel, dtype=np.float64) if full_qvel is not None else 0.0
        if "cube_qvel" in options: self.data.qvel[self._cube_qvel_adr : self._cube_qvel_adr+6] = np.asarray(options["cube_qvel"], dtype=np.float64)

        self.data.ctrl[:] = self._compose_ctrl_from_qpos()
        mujoco.mj_forward(self.model, self.data)
        settle = int(options.get("settle_steps",0))
        for _ in range(settle):
            mujoco.mj_step(self.model, self.data)
            self.data.ctrl[:] = self._compose_ctrl_from_qpos()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.data.ctrl[:] = np.clip(action, self.action_low, self.action_high)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._elapsed_steps +=1
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _normalize_quat(q): return q / np.linalg.norm(q)
    @staticmethod
    def _validate_initial_red_face(v): return v if v in {"down","random"} else "down"
    @staticmethod
    def _normalize_xy_jitter(j):
        arr = np.asarray(j, dtype=np.float64)
        return np.array([float(j),float(j)]) if arr.ndim==0 else arr
    @staticmethod
    def _quat_multiply(q1,q2):
        w1,x1,y1,z1=q1;w2,x2,y2,z2=q2
        return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dtype=np.float64)
    @staticmethod
    def _quat_from_axis_angle(axis,a):
        axis=axis/np.linalg.norm(axis);h=a/2
        return np.array([np.cos(h), axis[0]*np.sin(h), axis[1]*np.sin(h), axis[2]*np.sin(h)], dtype=np.float64)
    @classmethod
    def _sample_random_nonsolved_quaternion(cls,rng):
        candidates=[]
        for q in cls._axis_aligned_quaternions():
            if cls._red_face_up_alignment_for_quat(q)>=0.95:continue
            candidates.append(q)
        return candidates[int(rng.integers(len(candidates)))].copy()
    @classmethod
    def _axis_aligned_quaternions(cls):
        if not hasattr(cls,"_AXIS_ALIGNED_QUATERNIONS"):
            qs=[];seen=set()
            for rx in [0,np.pi/2,np.pi,np.pi*1.5]:
                for ry in [0,np.pi/2,np.pi,np.pi*1.5]:
                    for rz in [0,np.pi/2,np.pi,np.pi*1.5]:
                        q=cls._quat_multiply(cls._quat_from_axis_angle([0,0,1],rz),cls._quat_multiply(cls._quat_from_axis_angle([0,1,0],ry),cls._quat_from_axis_angle([1,0,0],rx)))
                        q=cls._normalize_quat(q)
                        if q[0]<0:q=-q
                        k=tuple(np.round(q,8))
                        if k not in seen:seen.add(k);qs.append(q)
            cls._AXIS_ALIGNED_QUATERNIONS=qs
        return [q.copy() for q in cls._AXIS_ALIGNED_QUATERNIONS]
    @classmethod
    def _red_face_up_alignment_for_quat(cls,q):
        q=cls._normalize_quat(q);w,x,y,z=q
        n=np.array([2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)],dtype=np.float64)
        return float(np.dot(n,cls.WORLD_UP))

# ====================== 【新增】奖励日志与绘图 ======================
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_episodes = 0
        self.reward_queue = deque(maxlen=50)

    def _on_step(self) -> bool:
        for env_idx in range(self.training_env.num_envs):
            infos = self.locals["infos"]
            if "episode" in infos[env_idx]:
                ep_r = infos[env_idx]["episode"]["r"]
                ep_l = infos[env_idx]["episode"]["l"]
                self.episode_rewards.append(ep_r)
                self.episode_lengths.append(ep_l)
                self.reward_queue.append(ep_r)
                self.total_episodes +=1
                if self.total_episodes %10 ==0:
                    avg=np.mean(self.reward_queue)
                    print(f"🎯 回合 {self.total_episodes:4d} | 奖励: {ep_r:6.2f} | 平均50轮: {avg:6.2f}")
        return True

    def plot_curve(self):
        plt.figure(figsize=(12,5))
        plt.plot(self.episode_rewards, alpha=0.4, label="每轮奖励", color="#55f")
        if len(self.episode_rewards)>=50:
            sm = np.convolve(self.episode_rewards, np.ones(50)/50, mode="valid")
            plt.plot(np.arange(49, len(self.episode_rewards)), sm, linewidth=2, color="#f44", label="50轮滑动平均")
        plt.xlabel("训练轮数")
        plt.ylabel("总奖励")
        plt.title("OrcaHand PPO 训练曲线")
        plt.legend()
        plt.grid(True)
        plt.savefig("orcahand_training_reward.png", dpi=300)
        plt.show()

# ====================== 【新增】训练函数 ======================
def train_ppo():
    # 多环境并行训练（无窗口）
    env = make_vec_env(lambda: OrcaHandRightCubeOrientation(render_mode=None, version="v1"), n_envs=2, seed=42)
    callback = RewardLoggerCallback()

    model_path = "orcahand_cube_ppo"
    if os.path.exists(f"{model_path}.zip"):
        print("✅ 加载已有模型继续训练...")
        model = PPO.load(model_path, env=env, verbose=1)
    else:
        print("❌ 新建PPO模型...")
        model = PPO(
            "MlpPolicy", env,
            policy_kwargs=dict(activation_fn=nn.LeakyReLU, net_arch=dict(pi=[256,128], vf=[256,128])),
            learning_rate=3e-4, batch_size=256, n_steps=1024, gamma=0.99, verbose=1
        )

    print("🚀 开始训练 OrcaHand 魔方翻转任务...")
    model.learn(total_timesteps=2_000_000, callback=callback, reset_num_timesteps=False)
    model.save(model_path)
    env.close()
    callback.plot_curve()
    print("✅ 训练完成！模型已保存：orcahand_cube_ppo.zip")

# ====================== 【新增】测试函数 ======================
def test_ppo():
    print("🎬 测试 PPO 策略（可视化窗口）")
    env = OrcaHandRightCubeOrientation(render_mode="human", version="v1")
    model = PPO.load("orcahand_cube_ppo_max500", env=env)

    for episode in range(1):
        obs, info = env.reset()
        total_reward = 0
        done = False
        step = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step +=1
            env.render()
            time.sleep(0.06)
        time.sleep(2)
        print(f"测试轮次 {episode+1} | 总奖励 {total_reward:.2f} | 步数 {step} | 成功情况: {info}")
    env.close()

# ====================== 主入口 ======================
if __name__ == "__main__":
    TRAIN_MODE = False   # 训练：True
                        # 测试：False
    if TRAIN_MODE:
        train_ppo()
    else:
        test_ppo()