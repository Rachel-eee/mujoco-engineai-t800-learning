"""打印 T800 机器人的所有连杆（body/link）信息。"""

import os

import mujoco

# 先切到脚本目录，避免 Windows 下含中文的绝对路径无法被 MuJoCo 打开
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path("t800.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)  # 更新 data.xpos，得到各连杆的世界坐标

print(f"=== T800 连杆(body)列表（共 {model.nbody} 个，含 world） ===")
print(f"{'ID':>3}  {'名称':<22}{'父ID':>4}  {'父连杆':<20}{'质量kg':>8}  x         y         z")
print("-" * 104)

for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "world"
    parent_id = model.body_parentid[i]
    parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) or "world"
    mass = model.body_mass[i]
    p = data.xpos[i]
    print(f"{i:>3}  {name:<22}{parent_id:>4}  {parent_name:<20}{mass:>8.3f}  "
          f"{p[0]:>8.4f}  {p[1]:>8.4f}  {p[2]:>8.4f}")
