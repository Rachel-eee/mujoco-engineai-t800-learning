"""打印 T800 所有 body 的 xpos（质心在世界坐标系下的位置）。"""

import os

import mujoco

# 先切到脚本目录，避免 Windows 下含中文的绝对路径无法被 MuJoCo 打开
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path("t800.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)  # 必须调用一次，xpos / xmat 等派生量才会被计算

print(f"=== T800 body xpos（共 {model.nbody} 个，世界坐标系，单位: m） ===")
print(f"{'ID':>3}  {'名称':<22}{'x':>10}{'y':>10}{'z':>10}")
print("-" * 58)

for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "world"
    xp = data.xpos[i]  # 该 body 的质心(COM)在世界坐标系下的位置
    print(f"{i:>3}  {name:<22}{xp[0]:>10.4f}{xp[1]:>10.4f}{xp[2]:>10.4f}")
