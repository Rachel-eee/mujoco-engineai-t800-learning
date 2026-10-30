from pathlib import Path
import os

import mujoco
import numpy as np


def count_world_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    n = 0
    for i in range(int(data.ncon)):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[int(c.geom1)])
        b2 = int(model.geom_bodyid[int(c.geom2)])
        if (b1 == 0) ^ (b2 == 0):
            n += 1
    return n


def main():
    # MuJoCo C++ 层在 Windows 下打不开含中文的绝对路径, 必须用相对路径
    os.chdir(Path(__file__).resolve().parent)
    xml_path = "t800.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    z_start = 0.50
    z_end = 1.50
    dz = 0.02

    zs = np.arange(z_start, z_end + 0.5 * dz, dz, dtype=np.float32)

    print(f"XML文件: {xml_path}")
    print(f"扫描范围: z={z_start:.2f}m ~ {z_end:.2f}m, step={dz:.2f}m, 点数={zs.size}")
    print()
    print(f"{'z(m)':>6}  {'world_contact_count':>20}  {'data.ncon':>8}")
    print("-" * 40)

    for z in zs:
        data.qpos[:] = 0
        data.qvel[:] = 0
        if model.nq >= 7:
            data.qpos[2] = float(z)
            data.qpos[3] = 1.0
            data.qpos[4] = 0.0
            data.qpos[5] = 0.0
            data.qpos[6] = 0.0
        else:
            raise RuntimeError(f"模型 nq={model.nq}，看起来不是 floating base，无法用 qpos[2] 扫描高度。")

        mujoco.mj_forward(model, data)

        world_contacts = count_world_contacts(model, data)
        print(f"{float(z):6.2f}  {world_contacts:20d}  {int(data.ncon):8d}")


if __name__ == "__main__":
    main()

