"""打印 T800 机器人关键状态参数：nq / nv / nu，以及 qpos / qvel / ctrl 的初始值。"""

import os

import mujoco
import numpy as np

# 先切到脚本目录，避免 Windows 下含中文的绝对路径无法被 MuJoCo 打开
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path("t800.xml")
data = mujoco.MjData(model)

print("=== T800 关键参数数量 ===")
print(f"nq   (广义坐标 qpos 数量) : {model.nq}")
print(f"nv   (广义速度 qvel 数量) : {model.nv}")
print(f"nu   (执行器 ctrl 数量)   : {model.nu}")

print("\n=== qpos 初始值 ===")
print(f"长度: {len(data.qpos)}")
print(f"qpos = {np.array2string(data.qpos, precision=4, separator=', ', max_line_width=200)}")
print("  注释: 前 7 个是 freejoint (xyz + 四元数 wxyz)，后 25 个是各铰链关节角")

print("\n=== qvel 初始值 ===")
print(f"长度: {len(data.qvel)}")
print(f"qvel = {np.array2string(data.qvel, precision=4, separator=', ', max_line_width=200)}")
print("  注释: 前 6 个是 freejoint 线速度 + 角速度 (vx,vy,vz,wx,wy,wz)，后 25 个是各铰链角速度")

print("\n=== ctrl 初始值 ===")
print(f"长度: {len(data.ctrl)}")
print(f"ctrl = {np.array2string(data.ctrl, precision=4, separator=', ', max_line_width=200)}")
print("  注释: 25 个电机的目标力矩/位置，默认为 0")

# 按关节索引打印 qpos 初始值（更直观）
print("\n=== 各关节 qpos 初始值（从关节索引看） ===")
print(f"{'ID':>3}  {'名称':<22}{'qpos索引':>9}{'qpos值':>10}")
print("-" * 48)
for i in range(model.njnt):
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "(未命名)"
    qadr = model.jnt_qposadr[i]
    if jname == "(未命名)" and i == 0:  # freejoint
        qname = f"freejoint qpos[{qadr}:{qadr+7}]"
        val = np.array2string(data.qpos[qadr:qadr+7], precision=4, separator=",")
        print(f"{i:>3}  {qname}")
        print(f"       取值: {val}")
    else:
        print(f"{i:>3}  {jname:<22}{qadr:>9}{data.qpos[qadr]:>10.4f}")
