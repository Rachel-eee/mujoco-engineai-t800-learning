"""用 MuJoCo viewer 把 T800 机器人展示到界面上。"""

import os
import time

import mujoco
import mujoco.viewer

# 先切到脚本目录，避免 Windows 下含中文的绝对路径无法被 MuJoCo 打开
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path("t800.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)  # 先算一次正运动学，保证首帧位姿正确

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 初始视角：看向机器人腰部，距离 3 m
    viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -15

    while viewer.is_running():
        # 仅做展示：不推进物理，机器人保持初始姿态
        # 若想让它在重力下运动，取消下面这行注释
        # mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(1 / 60)  # 约 60 FPS，避免空转占满 CPU

print("viewer 已关闭")
