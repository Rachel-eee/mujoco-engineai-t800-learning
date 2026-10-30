from pathlib import Path
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


# MuJoCo C++ 层在 Windows 下打不开含中文的绝对路径
os.chdir(Path(__file__).resolve().parent)

ROOT = Path(__file__).resolve().parent
LOCAL_MNN_PYTHON = ROOT / ".python_mnn"
if not LOCAL_MNN_PYTHON.exists():
    LOCAL_MNN_PYTHON = ROOT / "code" / ".python_mnn"
sys.path.insert(0, str(LOCAL_MNN_PYTHON))

import msvcrt
import MNN  # type: ignore


XML_FILE = "t800.xml"                                       # 相对路径
POLICY_FILE = "policy/t800_260318_150533_60000.mnn"         # 相对路径

BLOCK_SIZE = 0.04
BLOCK_POS = (0.80, 0.00, 0.12)

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
CMD_STEP = 0.05
CMD_MAX = 0.6


def build_scene_xml() -> str:
    return f"""
<mujoco model="t800_with_block">
    <include file="{XML_FILE}"/>
    <worldbody>
        <body name="red_block" pos="{BLOCK_POS[0]} {BLOCK_POS[1]} {BLOCK_POS[2]}">
            <freejoint/>
            <geom
                name="red_block_geom"
                type="box"
                size="{BLOCK_SIZE} {BLOCK_SIZE} {BLOCK_SIZE}"
                mass="0.2"
                rgba="1 0 0 1"
                friction="1.0 0.1 0.1"
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


def clamp_cmd(x: float) -> float:
    return -CMD_MAX if x < -CMD_MAX else CMD_MAX if x > CMD_MAX else x


def handle_keys(command: np.ndarray) -> bool:
    changed = False
    while msvcrt.kbhit():
        key = msvcrt.getwch()
        if not key:
            continue
        key = key.lower()
        if key in ("\x1b", "x"):
            return True
        if key == "w":
            command[0] = clamp_cmd(float(command[0] + CMD_STEP))
            changed = True
        elif key == "s":
            command[0] = clamp_cmd(float(command[0] - CMD_STEP))
            changed = True
        elif key == "a":
            command[1] = clamp_cmd(float(command[1] + CMD_STEP))
            changed = True
        elif key == "d":
            command[1] = clamp_cmd(float(command[1] - CMD_STEP))
            changed = True
        elif key == " ":
            command[:] = 0.0
            changed = True
    if changed:
        cmd = command.astype(np.float32)
        cmd_scaled = (cmd * COMMAND_SCALE).astype(np.float32)
        print("command:", cmd.round(3).tolist(), "scaled:", cmd_scaled.round(3).tolist())
    return False


def main():
    model = mujoco.MjModel.from_xml_string(build_scene_xml())
    data = mujoco.MjData(model)
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
    wall_start = time.time()

    print("keys: W/S 前后 | A/D 左右 | SPACE 停止 | X 或 ESC 退出")
    print("command:", command.astype(np.float32).round(3).tolist(), "scaled:", (command * COMMAND_SCALE).round(3).tolist())
    print(
        "block:",
        {
            "name": "red_block",
            "size_m": [0.08, 0.08, 0.08],
            "pos": [BLOCK_POS[0], BLOCK_POS[1], BLOCK_POS[2]],
        },
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < SIM_DURATION:
            if handle_keys(command):
                break

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
                next_control_t += CONTROL_DT

            mujoco.mj_step(model, data)
            viewer.sync()

            sleep_time = model.opt.timestep - (time.time() - wall_start - data.time)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()

