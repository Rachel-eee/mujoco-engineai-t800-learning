from pathlib import Path
import math
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


# MuJoCo 和 MNN 的 C++ 层在 Windows 下都打不开含中文的绝对路径,
# 所以必须先切到脚本目录, 再用相对路径加载
os.chdir(Path(__file__).resolve().parent)

ROOT = Path(__file__).resolve().parent
LOCAL_MNN_PYTHON = ROOT / ".python_mnn"
if not LOCAL_MNN_PYTHON.exists():
    LOCAL_MNN_PYTHON = ROOT / "code" / ".python_mnn"
sys.path.insert(0, str(LOCAL_MNN_PYTHON))

import MNN  # type: ignore


XML_FILE = "t800.xml"                                       # 相对路径
POLICY_FILE = "policy/t800_260318_150533_60000.mnn"         # 相对路径

ACTIVE_JOINT_NAMES = [
    "J00_HIP_PITCH_L",
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R",
    "J19_SHOULDER_ROLL_R",
    "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R",
    "J22_ELBOW_YAW_R",
]

DEFAULT_Q = np.array(
    [
        -0.06,
        0.0,
        0.0,
        0.12,
        -0.06,
        0.0,
        -0.06,
        0.0,
        0.0,
        0.12,
        -0.06,
        0.0,
        0.0,
        0.15,
        0.0,
        -0.25,
        0.0,
        0.0,
        -0.15,
        0.0,
        -0.25,
        0.0,
    ],
    dtype=np.float32,
)

HOMING_Q = np.array(
    [
        0.0,
        0.0,
        -0.12,
        0.24,
        -0.12,
        0.0,
        0.0,
        0.0,
        -0.12,
        0.24,
        -0.12,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)

ACTION_SCALE = np.array(
    [
        0.5,
        0.2,
        0.2,
        0.5,
        0.5,
        0.2,
        0.5,
        0.2,
        0.2,
        0.5,
        0.5,
        0.2,
        0.2,
        0.2,
        0.05,
        0.2,
        0.05,
        0.2,
        0.2,
        0.05,
        0.2,
        0.05,
    ],
    dtype=np.float32,
)

KP = np.array(
    [
        180,
        100,
        100,
        180,
        40,
        40,
        180,
        100,
        100,
        180,
        40,
        40,
        60,
        50,
        50,
        60,
        50,
        60,
        50,
        50,
        60,
        50,
    ],
    dtype=np.float32,
)

KD = np.array(
    [
        5.0,
        3.0,
        3.0,
        5.0,
        0.3,
        0.3,
        5.0,
        3.0,
        3.0,
        5.0,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
        0.3,
    ],
    dtype=np.float32,
)

COMMAND_SCALE = np.array([2.0, 2.0, 1.0], dtype=np.float32)
SIM_DURATION = 60.0


def joint_qpos_addrs(model):
    addrs = []
    for name in ACTIVE_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        addrs.append(model.jnt_qposadr[joint_id])
    return np.array(addrs, dtype=np.int32)


def joint_dof_addrs(model):
    addrs = []
    for name in ACTIVE_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        addrs.append(model.jnt_dofadr[joint_id])
    return np.array(addrs, dtype=np.int32)


def actuator_ids(model):
    ids = []
    for name in ACTIVE_JOINT_NAMES:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_" + name)
        ids.append(actuator_id)
    return np.array(ids, dtype=np.int32)


def quat_to_rot_wxyz(quat):
    w, x, y, z = quat / max(np.linalg.norm(quat), 1e-9)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def quat_to_rpy_deg(quat):
    w, x, y, z = quat
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [round(math.degrees(v), 2) for v in (roll, pitch, yaw)]


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def build_single_obs(model, data, last_action, q_addrs, v_addrs):
    q = data.qpos[q_addrs].astype(np.float32)
    qd = data.qvel[v_addrs].astype(np.float32)
    base_quat = data.qpos[3:7].astype(np.float32)
    base_angvel = data.qvel[3:6].astype(np.float32)
    gravity = -quat_to_rot_wxyz(base_quat).T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.concatenate([q - DEFAULT_Q, qd * 0.05, last_action, base_angvel, gravity]).astype(np.float32)


def infer_action(net, session, input_tensor, output_tensor, obs):
    host_input = MNN.Tensor(
        input_tensor.getShape(),
        MNN.Halide_Type_Float,
        obs.reshape(1, -1).astype(np.float32),
        MNN.Tensor_DimensionType_Caffe,
    )
    input_tensor.copyFrom(host_input)
    net.runSession(session)

    output_shape = output_tensor.getShape()
    host_output = MNN.Tensor(
        output_shape,
        MNN.Halide_Type_Float,
        np.zeros(output_shape, dtype=np.float32),
        MNN.Tensor_DimensionType_Caffe,
    )
    output_tensor.copyToHostTensor(host_output)
    return np.asarray(host_output.getData(), dtype=np.float32).reshape(-1)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_FILE))
    data = mujoco.MjData(model)
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = float(1.5)

    mujoco.mj_forward(model, data)

    q_addrs = joint_qpos_addrs(model)
    v_addrs = joint_dof_addrs(model)
    a_ids = actuator_ids(model)

    net = MNN.Interpreter(str(POLICY_FILE))
    session = net.createSession()
    input_tensor = net.getSessionInput(session)
    output_tensor = net.getSessionOutput(session)

    history = np.zeros((15, 72), dtype=np.float32)
    last_action = np.zeros(22, dtype=np.float32)
    command = np.array([0.4, 0.0, 0.0], dtype=np.float32)

    single_obs = build_single_obs(model, data, last_action, q_addrs, v_addrs)
    history[:] = single_obs

    next_control_t = 0.0
    next_log_t = 0.0
    wall_start = time.time()

    print(f"policy: {POLICY_FILE}")
    print(f"xml: {XML_FILE}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < SIM_DURATION:
            if data.time >= next_control_t - 1e-9:
                history[:-1] = history[1:]
                history[-1] = build_single_obs(model, data, last_action, q_addrs, v_addrs)

                obs = np.concatenate([history.reshape(-1), command * COMMAND_SCALE]).astype(np.float32)

                action = infer_action(net, session, input_tensor, output_tensor, obs)
                q_des = DEFAULT_Q + action * ACTION_SCALE

                if data.time < 0.5:
                    w = smoothstep(data.time / 0.5)
                    q_des = HOMING_Q * (1 - w) + q_des * w

                q = data.qpos[q_addrs].astype(np.float32)
                qd = data.qvel[v_addrs].astype(np.float32)
                tau = KP * (q_des - q) - KD * qd

                lo = model.actuator_ctrlrange[a_ids, 0]
                hi = model.actuator_ctrlrange[a_ids, 1]
                data.ctrl[a_ids] = np.clip(tau, lo, hi)

                last_action = action.copy()
                next_control_t += 0.01

            mujoco.mj_step(model, data)
            viewer.sync()

            pos = data.qpos[:3].astype(np.float32)
            rpy = np.array(quat_to_rpy_deg(data.qpos[3:7]), dtype=np.float32)

            if data.time >= next_log_t - 1e-9:
                print(
                    "t",
                    round(float(data.time), 2),
                    "pos",
                    pos.round(3).tolist(),
                    "rpy",
                    rpy.round(2).tolist(),
                    "cmd",
                    command.round(3).tolist(),
                )
                next_log_t += 0.5

            sleep_time = model.opt.timestep - (time.time() - wall_start - data.time)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
