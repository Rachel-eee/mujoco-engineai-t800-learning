from pathlib import Path
import math
import time

import mujoco
import mujoco.viewer


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def main():
    xml_path = Path(__file__).with_name("t800.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    qpos0 = data.qpos.copy()

    kp = 50.0
    kd = 2.0 * math.sqrt(kp)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"已加载: {xml_path.name}")
        print(f"PD保持初始位置: kp={kp}, kd={kd:.3f} (基于每个actuator对应的joint)")
        print("关闭窗口或 Ctrl+C 可结束。")

        while viewer.is_running():
            for aid in range(model.nu):
                jnt_id = int(model.actuator_trnid[aid, 0])
                if jnt_id < 0 or jnt_id >= model.njnt:
                    continue

                jnt_type = int(model.jnt_type[jnt_id])
                if jnt_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                    continue

                qposadr = int(model.jnt_qposadr[jnt_id])
                dofadr = int(model.jnt_dofadr[jnt_id])

                q = float(data.qpos[qposadr])
                qd = float(data.qvel[dofadr])
                q_des = float(qpos0[qposadr])

                u = kp * (q_des - q) - kd * qd

                if int(model.actuator_ctrllimited[aid]):
                    lo, hi = float(model.actuator_ctrlrange[aid, 0]), float(model.actuator_ctrlrange[aid, 1])
                    u = clamp(u, lo, hi)

                data.ctrl[aid] = u

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()

