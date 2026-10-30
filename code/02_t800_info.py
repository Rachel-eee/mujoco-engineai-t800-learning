"""加载 T800 模型并打印基本统计信息（body / joint / actuator 数量）。"""

import os
import mujoco

# MuJoCo 在 Windows 下用绝对路径打开含中文的目录会失败，
# 所以先切到脚本所在目录，再用相对路径加载（include 也会以该目录为基准解析）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path("t800.xml")

print("=== T800 模型信息 ===")
print(f"身体部件数量 (nbody) : {model.nbody}")
print(f"关节数量    (njnt)  : {model.njnt}")
print(f"电机数量    (nu)    : {model.nu}")
