"""打印 T800 所有 body 的 body_pos（相对父 body 的位置，来自 XML <body pos="...">）。"""

import os

import mujoco

# 先切到脚本目录，避免 Windows 下含中文的绝对路径无法被 MuJoCo 打开
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path("t800.xml")

print(f"=== T800 body_pos（共 {model.nbody} 个，含 world） ===")
print(f"{'ID':>3}  {'名称':<22}{'父body':<20}{'pos_x':>9}{'pos_y':>9}{'pos_z':>9}")
print("-" * 80)

for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "world"
    parent_id = model.body_parentid[i]
    parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) or "world"

    # body_pos: body 坐标系原点相对于父 body 坐标系的平移（来自 XML）
    bp = model.body_pos[i]

    print(f"{i:>3}  {name:<22}{parent_name:<20}{bp[0]:>9.4f}{bp[1]:>9.4f}{bp[2]:>9.4f}")
