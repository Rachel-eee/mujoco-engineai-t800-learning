from pathlib import Path
import math
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parent
LOCAL_MNN_PYTHON = ROOT / ".python_mnn"
if not LOCAL_MNN_PYTHON.exists():
    LOCAL_MNN_PYTHON = ROOT / "code" / ".python_mnn"
sys.path.insert(0, str(LOCAL_MNN_PYTHON))

import MNN  # type: ignore


XML_FILE = ROOT / "t800.xml"
POLICY_FILE = ROOT / "policy" / "t800_260318_150533_60000.mnn"

BLOCK_HALF_SIZE = 0.075
BLOCK_POS = (0.45, 0.00, 0.075)
BLOCK_MASS = 0.4
TORSO_JOINT_NAME = "J12_TORSO_YAW"
TORSO_ACTUATOR_NAME = "motor_J12_TORSO_YAW"
TORSO_KP = 80.0
TORSO_KD = 2.0

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
CONTROL_DT = 0.01
LOG_DT = 0.5


def build_scene_xml() -> str:
    xml_path = XML_FILE.as_posix()
    return f"""
<mujoco model="t800_kick_block">
    <include file="{xml_path}"/>
    <worldbody>
        <body name="red_block" pos="{BLOCK_POS[0]} {BLOCK_POS[1]} {BLOCK_POS[2]}">
            <freejoint/>
            <geom
                name="red_block_geom"
                type="box"
                size="{BLOCK_HALF_SIZE} {BLOCK_HALF_SIZE} {BLOCK_HALF_SIZE}"
                mass="{BLOCK_MASS}"
                rgba="1 0 0 1"
                friction="1.2 0.1 0.1"
            />
        </body>
    </worldbody>
</mujoco>
""".strip()


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


def single_actuator_info(model, joint_name: str, actuator_name: str):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if joint_id < 0:
        raise ValueError(f"找不到 joint: {joint_name}")
    if actuator_id < 0:
        raise ValueError(f"找不到 actuator: {actuator_name}")
    return {
        "joint_id": int(joint_id),
        "actuator_id": int(actuator_id),
        "qposadr": int(model.jnt_qposadr[joint_id]),
        "dofadr": int(model.jnt_dofadr[joint_id]),
        "lo": float(model.actuator_ctrlrange[actuator_id, 0]),
        "hi": float(model.actuator_ctrlrange[actuator_id, 1]),
    }


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


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def build_follow_block_command(data: mujoco.MjData, block_body_id: int) -> np.ndarray:
    base_pos = np.asarray(data.qpos[:3], dtype=np.float32)
    base_quat = np.asarray(data.qpos[3:7], dtype=np.float32)
    block_pos = np.asarray(data.xpos[block_body_id], dtype=np.float32)

    diff_world = block_pos - base_pos
    diff_world[2] = 0.0

    rot_world_from_base = quat_to_rot_wxyz(base_quat)
    diff_base = rot_world_from_base.T @ diff_world

    vx = clamp(1.2 * float(diff_base[0]), -0.2, 0.5)
    vy = clamp(2.0 * float(diff_base[1]), -0.3, 0.3)
    yaw = clamp(1.5 * math.atan2(float(diff_base[1]), max(float(diff_base[0]), 0.05)), -0.4, 0.4)
    return np.asarray([vx, vy, yaw], dtype=np.float32)


def main():
    model = mujoco.MjModel.from_xml_string(build_scene_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    q_addrs = joint_qpos_addrs(model)
    v_addrs = joint_dof_addrs(model)
    a_ids = actuator_ids(model)
    torso_info = single_actuator_info(model, TORSO_JOINT_NAME, TORSO_ACTUATOR_NAME)

    block_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")

    net = MNN.Interpreter(str(POLICY_FILE))
    session = net.createSession()
    input_tensor = net.getSessionInput(session)
    output_tensor = net.getSessionOutput(session)

    history = np.zeros((15, 72), dtype=np.float32)
    last_action = np.zeros(22, dtype=np.float32)
    single_obs = build_single_obs(model, data, last_action, q_addrs, v_addrs)
    history[:] = single_obs
    torso_q_ref = float(data.qpos[torso_info["qposadr"]])

    next_control_t = 0.0
    next_log_t = 0.0
    wall_start = time.time()

    print(f"policy: {POLICY_FILE.name}")
    print("demo: 机器人跟着木块方向前进，并持续踢木块")
    print(
        "block:",
        {
            "size_m": [0.15, 0.15, 0.15],
            "pos": [BLOCK_POS[0], BLOCK_POS[1], BLOCK_POS[2]],
            "mass": BLOCK_MASS,
        },
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < SIM_DURATION:
            if data.time >= next_control_t - 1e-9:
                history[:-1] = history[1:]
                history[-1] = build_single_obs(model, data, last_action, q_addrs, v_addrs)

                command = build_follow_block_command(data, block_body_id)
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

                torso_q = float(data.qpos[torso_info["qposadr"]])
                torso_qd = float(data.qvel[torso_info["dofadr"]])
                torso_tau = TORSO_KP * (torso_q_ref - torso_q) - TORSO_KD * torso_qd
                data.ctrl[torso_info["actuator_id"]] = clamp(torso_tau, torso_info["lo"], torso_info["hi"])

                last_action = action.copy()
                next_control_t += CONTROL_DT

            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time >= next_log_t - 1e-9:
                base_pos = data.qpos[:3].astype(np.float32)
                block_pos = np.asarray(data.xpos[block_body_id], dtype=np.float32)
                print(
                    "t",
                    round(float(data.time), 2),
                    "base_x",
                    round(float(base_pos[0]), 3),
                    "base_z",
                    round(float(base_pos[2]), 3),
                    "block_x",
                    round(float(block_pos[0]), 3),
                    "block_z",
                    round(float(block_pos[2]), 3),
                    "cmd",
                    command.round(3).tolist(),
                    "torso_q",
                    round(torso_q, 3),
                )
                next_log_t += LOG_DT

            sleep_time = model.opt.timestep - (time.time() - wall_start - data.time)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
