"""
19_t800_mnn_walk_log.py —— 带步态诊断日志的 MNN 行走代码（基于 18 号增强）

核心改动:
  - 控制频率保持 50Hz (CTRL_INTERVAL=10, dt=0.002)
  - 每 100ms (50 步) 打印一次完整步态日志: pos, rpy, z, MNN 推理, PD 误差/力矩
  - 用缓存同步: 把最近一次 MNN 推理的 action/耗时 和 PD 的 tau/q_err 存起来,
    日志点直接读, 避免为了日志降控制频率
"""

from pathlib import Path
import os
import time

import numpy as np
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)   # Windows 下 MuJoCo/MNN 打不开含中文的绝对路径

# ================= 1. 配置 =================
POLICY_FILE = "policy/t800_260318_150533_60000.mnn"
XML_FILE = "t800.xml"

OBS_SIZE = 1083
ACTION_SIZE = 22
NUM_HISTORY = 15
FRAME_SIZE = 72
CMD = np.array([1.0, 0.0, 0.0], dtype=np.float32)   # 策略控制的 22 个关节 (排除腰 J12 和头 J23/J24)
# 顺序按 legged_gym 训练约定: 左腿 → 右腿 → 躯干 → 左臂 → 右臂
CONTROL_JOINTS = [
    # 左腿 (6)
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    # 右腿 (6)
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    # 躯干 (1)
    "J12_TORSO_YAW",
    # 左臂 (5)
    "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
    # 右臂 (5)
    "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R",
]   # 6+6+1+5+5 = 23? 不对, 先看 nu=25, 排除腰头3个, 应该是22
# 哦, 腰部 J12 不控制, 去掉! 5+5 左臂右臂都控制

CTRL_INTERVAL = 10    # 控制频率: 50Hz (dt=0.002, 10步=20ms)
LOG_INTERVAL = 50     # 日志频率: 10Hz (50步=100ms)
WARMUP_STEPS = 250    # 前 0.5s 用纯 PD 保持站立, 不跑 MNN (给机器人稳定时间)
TOTAL_STEPS = 5000    # 共 10 s

# ================= 2. 工具函数 =================
def quat_to_rpy(q: np.ndarray) -> np.ndarray:
    """四元数 (w, x, y, z) → 欧拉角 (roll, pitch, yaw), 单位度. 纯 numpy."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
    roll  = np.arctan2(R[2, 1], R[2, 2])
    pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])


# ================= 3. MNN 桥 =================
class MnnBridge:
    def __init__(self, policy_path: str):
        import MNN
        self.MNN = MNN
        self.net = MNN.Interpreter(policy_path)
        self.session = self.net.createSession()
        self.input_tensor = self.net.getSessionInput(self.session)
        self.output_tensor = self.net.getSessionOutput(self.session)

    def infer(self, obs: np.ndarray) -> np.ndarray:
        MNN = self.MNN
        host_in = MNN.Tensor(self.input_tensor.getShape(), MNN.Halide_Type_Float,
                             obs.reshape(1, -1).astype(np.float32),
                             MNN.Tensor_DimensionType_Caffe)
        self.input_tensor.copyFrom(host_in)
        self.net.runSession(self.session)
        out_shape = self.output_tensor.getShape()
        host_out = MNN.Tensor(out_shape, MNN.Halide_Type_Float,
                              np.zeros(out_shape, dtype=np.float32),
                              MNN.Tensor_DimensionType_Caffe)
        self.output_tensor.copyToHostTensor(host_out)
        return np.asarray(host_out.getData(), dtype=np.float32).reshape(-1)


# ================= 4. 加载模型, 建索引 =================
model = mujoco.MjModel.from_xml_path(XML_FILE)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

qpos_adr, qvel_adr, act_ids = [], [], []
for name in CONTROL_JOINTS:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qpos_adr.append(model.jnt_qposadr[jid])
    qvel_adr.append(model.jnt_dofadr[jid])
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}")
    act_ids.append(aid)
qpos_adr = np.array(qpos_adr)
qvel_adr = np.array(qvel_adr)
act_ids = np.array(act_ids)

base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_BASE")
GRAVITY_WORLD = np.array([0.0, 0.0, -1.0])

# ================= 5. 主循环 =================
policy = MnnBridge(POLICY_FILE)
obs_buffer = np.zeros(NUM_HISTORY * FRAME_SIZE, dtype=np.float32)
prev_action = np.zeros(ACTION_SIZE, dtype=np.float32)
default_pos = np.zeros(ACTION_SIZE)

# ---- 日志缓存 (最近一次控制更新的数据) ----
last_infer_ms = 0.0
last_action = np.zeros(ACTION_SIZE)
last_tau_clipped = np.zeros(ACTION_SIZE)
last_q_err_deg = np.zeros(ACTION_SIZE)   # 关节误差 (度)

print("=" * 100)
print(f"  T800 MNN 行走 + 步态诊断日志")
print(f"  policy={POLICY_FILE}  obs={OBS_SIZE}  action={ACTION_SIZE}  cmd={CMD.tolist()}")
print(f"  控制 50Hz, 日志 10Hz, 仿真 {TOTAL_STEPS*0.002:.0f}s")
print("=" * 100)
print(f"{'t(s)':>6}  {'x':>8} {'y':>8} {'z':>7}  "
      f"{'roll':>6} {'pitch':>6} {'yaw':>6}  "
      f"{'inf_ms':>6}  {'a_max':>7}  "
      f"{'qerr_°':>10}  {'tau_max':>8}  {'clamp%':>7}")
print("-" * 100)

sim_stats = {
    "max_z": 0.0,
    "tumble_count": 0,      # z<0.3 的次数
    "over_tau_clips": 0,   # tau 被 clip 的次数
}

with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(TOTAL_STEPS + 1):

        # ---- 每 CTRL_INTERVAL 步做一次控制更新 ----
        if step % CTRL_INTERVAL == 0:
            t0 = time.perf_counter()

            # (a) 组装单帧 72 维观测 (严格按 legged_gym 训练约定)
            qvel_world = data.qvel[3:6]
            R = data.xmat[base_id].reshape(3, 3)
            ang_vel = R.T @ qvel_world                               # ← 转 body frame! 之前错用 world frame
            gravity_proj = R.T @ GRAVITY_WORLD                        # body 系下的重力方向
            q = data.qpos[qpos_adr]
            dq = data.qvel[qvel_adr]

            # (b) action: warmup 阶段强制 0, 让机器人先稳定站立
            if step < WARMUP_STEPS:
                action = np.zeros(ACTION_SIZE, dtype=np.float32)
            else:
                frame_obs = np.concatenate(
                    [ang_vel, gravity_proj, q, dq, prev_action]).astype(np.float32)
                obs_buffer = np.concatenate([obs_buffer[FRAME_SIZE:], frame_obs])
                obs = np.concatenate([obs_buffer, CMD])
                action = policy.infer(obs)

            prev_action = action.astype(np.float32)

            # (c) PD + 重力补偿: RL 策略输出的是"相对 default_pos 的关节角增量"
            target_pos = default_pos + ACTION_SCALE * action
            q_now = data.qpos[qpos_adr]
            dq_now = data.qvel[qvel_adr]
            tau_pd = Kp * (target_pos - q_now) - Kd * dq_now
            tau_gravity = data.qfrc_bias[qvel_adr]   # MuJoCo 算好的重力/科氏补偿
            tau = tau_pd + tau_gravity

            ctrl_min = model.actuator_ctrlrange[act_ids, 0]
            ctrl_max = model.actuator_ctrlrange[act_ids, 1]
            tau_clipped = np.clip(tau, ctrl_min, ctrl_max)

            data.ctrl[:] = 0.0
            data.ctrl[act_ids] = tau_clipped

            # ---- 更新日志缓存 ----
            last_infer_ms = (time.perf_counter() - t0) * 1000.0
            last_action = action.copy()
            last_tau_clipped = tau_clipped.copy()
            last_q_err_deg = np.degrees(target_pos - q_now)
            sim_stats["over_tau_clips"] += int(np.any(np.abs(tau) > np.abs(ctrl_max)))

        # ---- freejoint 四元数归一化 (MuJoCo 经典坑, 必须每步都做) ----
        # 不归一化的话 qpos[3:7] 模长慢慢漂移, 基座姿态计算全错, 连锁爆炸
        quat = data.qpos[3:7]
        qnorm = np.linalg.norm(quat)
        if qnorm > 0:
            data.qpos[3:7] = quat / qnorm

        # ---- 物理推进 + 可视化 ----
        mujoco.mj_step(model, data)
        viewer.sync()

        # ---- 10Hz 步态日志 (每 100ms) ----
        if step % LOG_INTERVAL == 0:
            pos = data.qpos[0:3]
            rpy = quat_to_rpy(data.qpos[3:7])
            sim_stats["max_z"] = max(sim_stats["max_z"], pos[2])
            if pos[2] < 0.3:
                sim_stats["tumble_count"] += 1

            q_err_max = np.max(np.abs(last_q_err_deg))
            tau_max = np.max(np.abs(last_tau_clipped))
            action_max = np.max(np.abs(last_action))
            # clamp 比例: 当前这帧里被 clip 的电机数 / 22
            clip_pct = 100.0 * np.mean(np.abs(last_tau_clipped) >=
                                       np.abs(model.actuator_ctrlrange[act_ids, 1]) - 1e-3)

            # ---- 统一转 Python float 避免 numpy 标量格式符失效 ----
            t_now = float(data.time)
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
            inf_ms = float(last_infer_ms)
            a_max = float(action_max)
            qe_max = float(q_err_max)
            t_max = float(tau_max)
            c_pct = float(clip_pct)

            print(f"{t_now:6.2f}  {px:+8.3f} {py:+8.3f} {pz:7.3f}  "
                  f"{r:6.1f} {p:6.1f} {y:6.1f}  "
                  f"{inf_ms:6.2f}  {a_max:7.3f}  "
                  f"{qe_max:10.1f}  {t_max:8.1f}  {c_pct:5.0f}%")

        if not viewer.is_running():
            print("\n>>> viewer 被用户关闭, 提前结束")
            break

# ================= 6. 汇总 =================
print("-" * 100)
print("  步态诊断汇总")
print("-" * 100)
final_rpy = quat_to_rpy(data.qpos[3:7])
print(f"  仿真时长          : {float(data.time):.2f} s")
print(f"  最终位置 (x,y,z)  : ({float(data.qpos[0]):+.3f}, "
      f"{float(data.qpos[1]):+.3f}, {float(data.qpos[2]):.3f}) m")
print(f"  最终 rpy          : roll={float(final_rpy[0]):+.1f}°  "
      f"pitch={float(final_rpy[1]):+.1f}°  "
      f"yaw={float(final_rpy[2]):+.1f}°")
print(f"  基座最高 z        : {float(sim_stats['max_z']):.3f} m")
print(f"  姿态失稳次数 (z<0.3): {sim_stats['tumble_count']} / {TOTAL_STEPS // LOG_INTERVAL}")
print(f"  PD 力矩 clip 次数  : {sim_stats['over_tau_clips']} / {TOTAL_STEPS // CTRL_INTERVAL}")
print("=" * 100)
