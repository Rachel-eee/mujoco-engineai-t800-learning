from pathlib import Path
import math
import time

import mujoco
import mujoco.viewer


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def mat_to_rpy_zyx(r00: float, r01: float, r02: float, r10: float, r11: float, r12: float, r20: float, r21: float, r22: float):
    pitch = math.asin(clamp(-r20, -1.0, 1.0))
    cp = math.cos(pitch)
    if abs(cp) < 1e-9:
        roll = 0.0
        yaw = math.atan2(-r01, r11)
    else:
        roll = math.atan2(r21, r22)
        yaw = math.atan2(r10, r00)
    return roll, pitch, yaw


def find_base_body_id(model: mujoco.MjModel) -> int:
    for name in ("LINK_BASE", "base", "BASE", "pelvis", "root"):
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                return int(bid)
        except Exception:
            pass
    return 1 if model.nbody > 1 else 0


def main():
    xml_path = Path(__file__).with_name("t800.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    base_id = find_base_body_id(model)
    base_name = model.body(base_id).name or ("world" if base_id == 0 else f"<unnamed_{base_id}>")

    print_interval = 0.1
    duration = 10.0
    next_print_t = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"已加载: {xml_path.name}")
        print(f"基准body: id={base_id}, name={base_name}")
        print("每隔100ms打印一次 pos/rpy，持续10秒。关闭窗口或 Ctrl+C 可结束。")

        while viewer.is_running() and data.time < duration:
            if data.time + 1e-12 >= next_print_t:
                pos = data.xpos[base_id]
                m = data.xmat[base_id]
                roll, pitch, yaw = mat_to_rpy_zyx(
                    float(m[0]),
                    float(m[1]),
                    float(m[2]),
                    float(m[3]),
                    float(m[4]),
                    float(m[5]),
                    float(m[6]),
                    float(m[7]),
                    float(m[8]),
                )
                print(
                    f"t={data.time:6.3f}s | "
                    f"pos=({pos[0]: .4f},{pos[1]: .4f},{pos[2]: .4f}) | "
                    f"rpy(rad)=({roll: .4f},{pitch: .4f},{yaw: .4f})"
                )
                next_print_t += print_interval

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()

