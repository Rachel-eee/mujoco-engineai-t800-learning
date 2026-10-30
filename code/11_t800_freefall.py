"""
T800 仿真演示：无电机力矩（ctrl=0）下机器人在重力作用下倒地。
每隔 100 ms 打印一次基座的 pos (xyz) 与 rpy (Roll-Pitch-Yaw, 度)，共 10 s。
"""

import os
import time
import numpy as np

import mujoco
import mujoco.viewer

# 先切到脚本目录，避免 Windows 下含中文的绝对路径无法被 MuJoCo 打开
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path("t800.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# ---- 仿真参数 ----
DT = model.opt.timestep                 # 0.002 s
PRINT_INTERVAL = 0.1                    # 100 ms
TOTAL_TIME = 10.0                       # 10 s
interval_steps = round(PRINT_INTERVAL / DT)  # 50 步
total_steps = round(TOTAL_TIME / DT)     # 5000 步

# 关节执行器默认零力矩（ctrl 全 0），所以不主动给力


def quat_to_rpy_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo 四元数 (w,x,y,z) → 欧拉角 (roll, pitch, yaw) 度（纯 numpy，无 C API）。"""
    w, x, y, z = quat_wxyz
    # 构造旋转矩阵 R = I + 2w*[v]x + 2*[v]x^2, 其中 v=(x,y,z), [v]x 为反对称矩阵
    # 简化公式 (w,x,y,z) 归一化后
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z

    R = np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz),  yz - wx],
        [xz - wy,         yz + wx,          1.0 - (xx + yy)],
    ])

    # ZYX 欧拉角 (yaw-pitch-roll)
    sy = np.clip(-R[2, 0], -1.0, 1.0)
    pitch = np.arcsin(sy)
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])


print("=== T800 零力矩仿真 ===")
print(f"dt={DT}s, 每 {interval_steps} 步打印一次 ({PRINT_INTERVAL*1000:.0f}ms), 共 {total_steps} 步 ({TOTAL_TIME}s)")
print(f"{'t(s)':>8}  {'x':>8}{'y':>8}{'z':>8}  |  {'roll':>8}{'pitch':>8}{'yaw':>8}")
print("-" * 80)

start_t = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -10

    for step in range(total_steps + 1):
        # 打印节拍：每隔 interval_steps 步
        if step % interval_steps == 0:
            pos = data.qpos[0:3]            # 基座世界坐标 (x,y,z)
            quat = data.qpos[3:7]           # 四元数 w,x,y,z
            rpy = quat_to_rpy_deg(quat)
            print(f"{data.time:>8.2f}  "
                  f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}  |  "
                  f"{rpy[0]:>8.2f}{rpy[1]:>8.2f}{rpy[2]:>8.2f}")

        mujoco.mj_step(model, data)
        viewer.sync()

        # 关掉窗口提前结束
        if not viewer.is_running():
            print("\nviewer 被关闭，提前停止")
            break

elapsed = time.time() - start_t
print("-" * 80)
print(f"仿真结束，墙钟耗时 {elapsed:.2f}s")
