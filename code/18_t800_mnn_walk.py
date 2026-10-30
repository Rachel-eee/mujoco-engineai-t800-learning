"""
18_t800_mnn_walk.py —— MNN 神经网络控制 T800 行走（最小教学版）

数据流:
    MuJoCo 状态 → 组装 obs (1083维) → MNN 推理 → action (22维)
    → PD 控制器算力矩 → mj_step → 机器人行走

obs 结构 (1083 = 72*15 + 3):
    - 每帧 72 维: 角速度(3) + 重力投影(3) + 关节位置(22) + 关节速度(22) + 上一帧action(22)
    - 历史堆叠 15 帧
    - 末尾 3 维: 速度指令 [前进速度, 侧向速度, 转向速度]

action → 力矩: 标准 RL 公式
    tau = Kp * (default_pos + action*SCALE - q) - Kd * dq
"""

from pathlib import Path
import os

import numpy as np
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)   # MuJoCo 在 Windows 下打不开含中文的绝对路径, 切到脚本目录用相对路径

# ================= 1. 基本配置 =================
# 注意: MuJoCo 和 MNN 的 C++ 层都打不开含中文的绝对路径, 所以全部用相对路径
POLICY_FILE = "policy/t800_260318_150533_60000.mnn"
XML_FILE = "t800.xml"

OBS_SIZE = 1083          # 网络输入
ACTION_SIZE = 22         # 网络输出
NUM_HISTORY = 15         # obs 历史堆叠帧数
FRAME_SIZE = 72          # 每帧观测维度
CMD = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # 速度指令: 向前 1 m/s, 不侧移, 不转向

# PD 增益 (与训练时一致的数量级)
Kp = 300.0
Kd = 10.0
ACTION_SCALE = 0.25      # action 缩放: 神经网络输出是 -1~1 附近的小值

# 策略控制的 22 个关节 (排除腰部 J12 和头部 J23/J24, 共 25-3=22)
CONTROL_JOINTS = [f"J{i:02d}_" + part for i, part in [
    (0, "HIP_PITCH_L"), (1, "HIP_ROLL_L"), (2, "HIP_YAW_L"),
    (3, "KNEE_PITCH_L"), (4, "ANKLE_PITCH_L"), (5, "ANKLE_ROLL_L"),
    (6, "HIP_PITCH_R"), (7, "HIP_ROLL_R"), (8, "HIP_YAW_R"),
    (9, "KNEE_PITCH_R"), (10, "ANKLE_PITCH_R"), (11, "ANKLE_ROLL_R"),
    (13, "SHOULDER_PITCH_L"), (14, "SHOULDER_ROLL_L"), (15, "SHOULDER_YAW_L"),
    (16, "ELBOW_PITCH_L"), (17, "ELBOW_YAW_L"),
    (18, "SHOULDER_PITCH_R"), (19, "SHOULDER_ROLL_R"), (20, "SHOULDER_YAW_R"),
    (21, "ELBOW_PITCH_R"), (22, "ELBOW_YAW_R"),
]]
# 未控制的 3 个执行器 (腰 + 头) ctrl 置 0
ALL_ACTUATOR_NAMES = [f"motor_{n}" for n in
                      [f"J{i:02d}_" + p for i, p in [
                          (0, "HIP_PITCH_L"), (1, "HIP_ROLL_L"), (2, "HIP_YAW_L"),
                          (3, "KNEE_PITCH_L"), (4, "ANKLE_PITCH_L"), (5, "ANKLE_ROLL_L"),
                          (6, "HIP_PITCH_R"), (7, "HIP_ROLL_R"), (8, "HIP_YAW_R"),
                          (9, "KNEE_PITCH_R"), (10, "ANKLE_PITCH_R"), (11, "ANKLE_ROLL_R"),
                          (12, "TORSO_YAW"),
                          (13, "SHOULDER_PITCH_L"), (14, "SHOULDER_ROLL_L"), (15, "SHOULDER_YAW_L"),
                          (16, "ELBOW_PITCH_L"), (17, "ELBOW_YAW_L"),
                          (18, "SHOULDER_PITCH_R"), (19, "SHOULDER_ROLL_R"), (20, "SHOULDER_YAW_R"),
                          (21, "ELBOW_PITCH_R"), (22, "ELBOW_YAW_R"),
                          (23, "HEAD_PITCH"), (24, "HEAD_YAW"),
                      ]]]

# ================= 2. MNN 桥 (复用 17 的推理逻辑, 精简版) =================
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


# ================= 3. 加载模型, 建立索引 =================
model = mujoco.MjModel.from_xml_path(XML_FILE)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# 22 个控制关节的 qpos/qvel 地址 和 对应执行器 id
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

# 基座 body id (用于读取角速度和重力投影)
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_BASE")
GRAVITY_WORLD = np.array([0.0, 0.0, -1.0])   # 单位重力方向

# ================= 4. 主循环 =================
policy = MnnBridge(POLICY_FILE)
obs_buffer = np.zeros(NUM_HISTORY * FRAME_SIZE, dtype=np.float32)  # 15帧历史
prev_action = np.zeros(ACTION_SIZE, dtype=np.float32)
default_pos = np.zeros(ACTION_SIZE)   # 目标姿态 = 初始全 0

CTRL_INTERVAL = 10    # 每 10 步控制一次: dt=0.002 → 50Hz
TOTAL_STEPS = 5000    # 共 10 s

print("=== T800 MNN 行走 ===")
print(f"policy: {POLICY_FILE}")
print(f"obs={OBS_SIZE}, action={ACTION_SIZE}, cmd={CMD.tolist()}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(TOTAL_STEPS + 1):

        # ---- 每 10 步做一次网络推理 ----
        if step % CTRL_INTERVAL == 0:
            # (a) 组装单帧 72 维观测
            ang_vel = data.qvel[3:6]                            # 基座角速度 (世界系)
            R = data.xmat[base_id].reshape(3, 3)
            gravity_proj = R.T @ GRAVITY_WORLD                  # 重力方向投影到机体系
            q = data.qpos[qpos_adr]                             # 22 关节位置
            dq = data.qvel[qvel_adr]                            # 22 关节速度
            frame_obs = np.concatenate([ang_vel, gravity_proj, q, dq, prev_action]).astype(np.float32)

            # (b) 历史堆叠: 滚动窗口 (最早的帧丢掉, 新帧放末尾)
            obs_buffer = np.concatenate([obs_buffer[FRAME_SIZE:], frame_obs])

            # (c) 拼完整 obs = 15帧历史(1080) + 指令(3)
            obs = np.concatenate([obs_buffer, CMD])

            # (d) 网络推理
            action = policy.infer(obs)
            prev_action = action.astype(np.float32)

            # (e) PD 控制器: action → 目标关节角 → 力矩
            target_pos = default_pos + ACTION_SCALE * action
            q_now = data.qpos[qpos_adr]
            dq_now = data.qvel[qvel_adr]
            tau = Kp * (target_pos - q_now) - Kd * dq_now

            # 写入 ctrl (先清零 3 个未控制电机, 再填 22 个)
            data.ctrl[:] = 0.0
            data.ctrl[act_ids] = np.clip(tau,
                                         model.actuator_ctrlrange[act_ids, 0],
                                         model.actuator_ctrlrange[act_ids, 1])

        # ---- 物理推进 + 可视化 ----
        mujoco.mj_step(model, data)
        viewer.sync()

        # 每 1s 打印一次行进状态
        if step % 500 == 0:
            base_z = data.qpos[2]
            print(f"t={data.time:5.2f}s  base_pos=({data.qpos[0]:+.3f}, "
                  f"{data.qpos[1]:+.3f}, {base_z:.3f})m")

        if not viewer.is_running():
            break

print(f"结束: 机器人 x 方向前进 {data.qpos[0]:+.3f} m")
