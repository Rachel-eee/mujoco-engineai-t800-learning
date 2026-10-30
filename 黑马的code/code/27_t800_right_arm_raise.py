from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

import msvcrt


ROOT = Path(__file__).resolve().parent
XML_FILE = ROOT / "t800.xml"

SIM_DURATION = 60.0
CONTROL_DT = 0.01
ARM_BLEND_TIME = 0.8

ALL_JOINT_NAMES = [
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
    "J12_TORSO_YAW",
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
    "J23_HEAD_PITCH",
    "J24_HEAD_YAW",
]

ARM_JOINT_NAMES = [
    "J18_SHOULDER_PITCH_R",
    "J19_SHOULDER_ROLL_R",
    "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R",
    "J22_ELBOW_YAW_R",
]

RAISE_DELTA = {
    "J18_SHOULDER_PITCH_R": -1.00,
    "J19_SHOULDER_ROLL_R": -0.35,
    "J20_SHOULDER_YAW_R": 0.10,
    "J21_ELBOW_PITCH_R": -0.45,
    "J22_ELBOW_YAW_R": 0.00,
}

KP_DEFAULT = 80.0
KD_DEFAULT = 3.0
KP_ARM = 100.0
KD_ARM = 4.0


def get_joint_infos(model: mujoco.MjModel):
    infos = []
    for joint_name in ALL_JOINT_NAMES:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_" + joint_name)
        if jnt_id < 0 or act_id < 0:
            continue
        infos.append(
            {
                "joint_name": joint_name,
                "joint_id": int(jnt_id),
                "act_id": int(act_id),
                "qposadr": int(model.jnt_qposadr[jnt_id]),
                "dofadr": int(model.jnt_dofadr[jnt_id]),
                "lo": float(model.actuator_ctrlrange[act_id, 0]),
                "hi": float(model.actuator_ctrlrange[act_id, 1]),
            }
        )
    return infos


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def smoothstep(x: float) -> float:
    x = 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
    return x * x * (3.0 - 2.0 * x)


def handle_keys(target_mode: str) -> str:
    while msvcrt.kbhit():
        key = msvcrt.getwch()
        if not key:
            continue
        key = key.lower()
        if key in ("\x1b", "x"):
            return "quit"
        if key == "w":
            print("mode: raise")
            return "raise"
        if key == "s":
            print("mode: lower")
            return "lower"
    return target_mode


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_FILE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    joint_infos = get_joint_infos(model)

    base_qpos_ref = data.qpos[:7].copy()
    q_ref = {info["joint_name"]: float(data.qpos[info["qposadr"]]) for info in joint_infos}
    q_raise = dict(q_ref)
    for joint_name, delta in RAISE_DELTA.items():
        if joint_name in q_raise:
            q_raise[joint_name] = q_ref[joint_name] + float(delta)

    target_mode = "lower"
    target_alpha = 0.0
    current_alpha = 0.0
    next_control_t = 0.0
    wall_start = time.time()

    print("demo: 固定 base，只控制右臂")
    print("keys: W 抬起右臂 | S 放下右臂 | X 或 ESC 退出")
    print("mode: lower")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < SIM_DURATION:
            target_mode = handle_keys(target_mode)
            if target_mode == "quit":
                break
            target_alpha = 1.0 if target_mode == "raise" else 0.0

            if data.time >= next_control_t - 1e-9:
                # 固定 floating base，让机器人始终直立展示
                data.qpos[:7] = base_qpos_ref
                data.qvel[:6] = 0.0

                step_alpha = CONTROL_DT / ARM_BLEND_TIME
                if current_alpha < target_alpha:
                    current_alpha = min(target_alpha, current_alpha + step_alpha)
                elif current_alpha > target_alpha:
                    current_alpha = max(target_alpha, current_alpha - step_alpha)
                arm_blend = smoothstep(current_alpha)

                for info in joint_infos:
                    name = info["joint_name"]
                    q = float(data.qpos[info["qposadr"]])
                    qd = float(data.qvel[info["dofadr"]])

                    if name in ARM_JOINT_NAMES:
                        q_des = (1.0 - arm_blend) * q_ref[name] + arm_blend * q_raise[name]
                        kp = KP_ARM
                        kd = KD_ARM
                    else:
                        q_des = q_ref[name]
                        kp = KP_DEFAULT
                        kd = KD_DEFAULT

                    tau = kp * (q_des - q) - kd * qd
                    data.ctrl[info["act_id"]] = clamp(tau, info["lo"], info["hi"])

                next_control_t += CONTROL_DT

            mujoco.mj_step(model, data)
            viewer.sync()

            sleep_time = model.opt.timestep - (time.time() - wall_start - data.time)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
