"""打印 T800 所有执行器（电机）的 ctrlrange。"""

import os
import mujoco

# 先切到脚本目录，避免 Windows 下含中文的绝对路径无法被 MuJoCo 打开
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path("t800.xml")

print(f"=== T800 执行器 ctrlrange（共 {model.nu} 个） ===")
print(f"{'ID':>3}  {'执行器名称':<28}{'类型':<10}{'关节':<22}{'ctrlrange_min':>14}{'ctrlrange_max':>14}{'限幅':>6}")
print("-" * 100)

for i in range(model.nu):
    act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"(act_{i})"

    # actuator 关联的 joint（T800 都是 <motor joint="..." />）
    joint_id = model.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "(无)"

    # actuator 类型
    a_type = "motor"
    if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_INTEGRATOR:
        a_type = "velocity"

    # ctrlrange
    ctrllimited = bool(model.actuator_ctrllimited[i])
    if ctrllimited:
        cmin = model.actuator_ctrlrange[i, 0]
        cmax = model.actuator_ctrlrange[i, 1]
        limited_str = "是"
    else:
        cmin = float("nan")
        cmax = float("nan")
        limited_str = "否"

    print(f"{i:>3}  {act_name:<28}{a_type:<10}{joint_name:<22}"
          f"{cmin:>14.1f}{cmax:>14.1f}{limited_str:>6}")

print("-" * 100)
print("注: 执行器在 serial_actuators.xml 里用的是 <motor>, gear=1，单位是 N·m（力矩模式）")
