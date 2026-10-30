"""打印 T800 模型的自由度参数：nq / nv / nu。"""

import os
import mujoco

# 先切到脚本目录，避免 Windows 下含中文的绝对路径无法被 MuJoCo 打开
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path("t800.xml")

print("=== T800 自由度信息 ===")
print(f"nq (广义坐标数量)    : {model.nq}")
print(f"nv (广义速度数量)    : {model.nv}")
print(f"nu (电机/执行器数量) : {model.nu}")
