"""
T800 PD 控制器示例：让 25 个 hinge 关节保持初始位置。
- 为了让 "基座不倒"，这里动态去掉 <freejoint/>，让 LINK_BASE 直接固定在 worldbody 上
  （不修改原始 XML，仅 Python 里字符串替换）
- PD 控制: tau = Kp * (q_target - q) + Kd * (dq_target - dq) + gravity_comp
- 重力补偿用 data.qfrc_bias（fixed base 时直接从 dof[0] 开始）
- 可视化 viewer + 每 200ms 打印一次基座高度 & 最大关节误差
"""

import os
import time
import numpy as np

import mujoco
import mujoco.viewer

# ---- 切到脚本目录，避免 Windows 下含中文的绝对路径无法被 MuJoCo 打开 ----
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---- 动态把 <freejoint/> 去掉，让基座固定 ----
with open("t800.xml", "r", encoding="utf-8") as f:
    main_xml = f.read()
# serial_t800.xml 里才是真正的 <freejoint/> 所在位置
with open("robot/t800/xml/serial_t800.xml", "r", encoding="utf-8") as f:
    t800_xml = f.read()
# 用字符串替换：注释掉 freejoint，保留那一行以便定位
t800_xml_fixed = t800_xml.replace("<freejoint/>", "<!-- <freejoint/>  (PD 控制演示里注释掉，固定基座) -->")
# 直接加载修改后的 t800.xml（包含 include，所以它引用 serial_t800.xml 时我们得……）
# 更简单：重建一个临时 xml，把 serial_t800.xml 也替换进去
main_xml_fixed = main_xml.replace(
    '<include file="robot/t800/xml/serial_t800.xml"/>',
    '<include file="robot/t800/xml/serial_t800_fixed.xml"/>',
)
# 把修改过的写一个临时文件（避免改原文件）
fixed_serial_path = "robot/t800/xml/serial_t800_fixed.xml"
with open(fixed_serial_path, "w", encoding="utf-8") as f:
    f.write(t800_xml_fixed)

# ---- 加载 fixed base 模型 ----
model = mujoco.MjModel.from_xml_string(main_xml_fixed)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print(f"=== fixed base 模型统计 ===")
print(f"nbody={model.nbody}, njnt={model.njnt}, nu={model.nu}")
print(f"nq={model.nq}, nv={model.nv}")

# ---- 关节 / 电机 索引准备 ----
# 所有关节都是 hinge（没有 freejoint 了）
hinge_jnt_ids = list(range(model.njnt))
qpos_adr = [model.jnt_qposadr[i] for i in hinge_jnt_ids]   # 长度 25
qvel_adr = [model.jnt_dofadr[i] for i in hinge_jnt_ids]    # 长度 25

# ---- 目标位置（初始关节角，全 0）----
q_target = np.array([data.qpos[adr] for adr in qpos_adr], dtype=np.float64)
dq_target = np.zeros(len(q_target))
print(f"q_target 长度: {len(q_target)}, 初始值: {q_target[:6]} ...")

# ---- PD 增益 ----
Kp = np.full(len(q_target), 400.0)   # 硬一点
Kd = np.full(len(q_target), 20.0)

# ---- 电机输出限幅 ----
CTRL_CLIP_MIN = model.actuator_ctrlrange[:, 0].copy()
CTRL_CLIP_MAX = model.actuator_ctrlrange[:, 1].copy()
print(f"ctrlrange 样本 (前6对):")
for i in range(min(6, model.nu)):
    print(f"  电机{i}: [{CTRL_CLIP_MIN[i]}, {CTRL_CLIP_MAX[i]}]")


def compute_pd_control() -> tuple[np.ndarray, np.ndarray]:
    """计算 25 个电机的目标力矩 + 关节误差。"""
    q_now = np.array([data.qpos[adr] for adr in qpos_adr])
    dq_now = np.array([data.qvel[adr] for adr in qvel_adr])

    err = q_target - q_now
    derr = dq_target - dq_now

    tau = Kp * err + Kd * derr

    # 重力补偿: fixed base 下 qfrc_bias 直接就是 hinge 对应的 DOF
    tau_bias = data.qfrc_bias[0: len(q_target)]
    tau += tau_bias

    tau = np.clip(tau, CTRL_CLIP_MIN, CTRL_CLIP_MAX)
    return tau, err


# ---- 仿真参数 ----
DT = model.opt.timestep
PRINT_INTERVAL = 0.2    # 200 ms
TOTAL_TIME = 15.0       # 15 s
interval_steps = round(PRINT_INTERVAL / DT)
total_steps = round(TOTAL_TIME / DT)

# 基座 body id（fixed base 下基座是 LINK_BASE）
base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_BASE")

print(f"\n=== T800 PD 控制演示 (fixed base) ===")
print(f"dt={DT}s, 每 {interval_steps} 步打印一次 ({PRINT_INTERVAL*1000:.0f}ms), 共 {total_steps} 步 ({TOTAL_TIME}s)")
print(f"Kp={Kp[0]:.0f}, Kd={Kd[0]:.1f}, 控制关节数={len(q_target)}")
print(f"{'t(s)':>8}  {'基座z':>8}  {'最大关节误差(°)':>16}  {'平均关节误差(°)':>16}")
print("-" * 60)

start_t = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
    viewer.cam.distance = 3.5
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -15

    for step in range(total_steps + 1):
        ctrl, err = compute_pd_control()
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        viewer.sync()

        if step % interval_steps == 0:
            base_z = data.xpos[base_body_id][2]   # fixed base 下基座位置从 xpos 读
            err_deg = np.degrees(err)
            print(f"{data.time:>8.2f}  {base_z:>8.3f}  "
                  f"{np.max(np.abs(err_deg)):>16.3f}  {np.mean(np.abs(err_deg)):>16.4f}")

        if not viewer.is_running():
            print("\nviewer 被关闭，提前停止")
            break

elapsed = time.time() - start_t
print("-" * 60)
print(f"仿真结束，墙钟耗时 {elapsed:.2f}s")

# ---- 清理临时文件 ----
try:
    os.remove(fixed_serial_path)
except OSError:
    pass

# 最终状态打印
q_now_final = np.array([data.qpos[adr] for adr in qpos_adr])
err_final_deg = np.degrees(q_target - q_now_final)
base_z_final = data.xpos[base_body_id][2]
print(f"\n=== 最终状态 ===")
print(f"  基座 z:        {base_z_final:.4f} m")
print(f"  最大关节误差:  {np.max(np.abs(err_final_deg)):.3f} 度")
print(f"  平均关节误差:  {np.mean(np.abs(err_final_deg)):.4f} 度")
