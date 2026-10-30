"""
验证 MuJoCo 执行器的 ctrl clamp 是否生效：
把所有电机的 data.ctrl 都设成 10000，
然后对比：原始输入 → data.ctrl (MuJoCo 内部 clip 后) → data.actuator_force (实际输出力矩)
"""

import os
import numpy as np
import mujoco

# 先切到脚本目录，避免 Windows 下含中文的绝对路径无法被 MuJoCo 打开
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path("t800.xml")
data = mujoco.MjData(model)

# ---- 1. 先看每个电机的 ctrlrange ----
print(f"=== T800 执行器 ctrlrange（共 {model.nu} 个） ===\n")

cmin = model.actuator_ctrlrange[:, 0].copy()
cmax = model.actuator_ctrlrange[:, 1].copy()
clipped_flags = [bool(model.actuator_ctrllimited[i]) for i in range(model.nu)]

for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}"
    limited = clipped_flags[i]
    c_range = f"[{cmin[i]:.1f}, {cmax[i]:.1f}]" if limited else "未限幅"
    print(f"  [{i:>2}] {name:<28} ctrlrange={c_range:<18}  ctrllimited={limited}")

# ---- 2. 暴力塞 10000 进去 ----
INPUT_VAL = 10000.0
data.ctrl[:] = INPUT_VAL
print(f"\n>>> 把 data.ctrl[:] 全部设成 {INPUT_VAL}")
print(f"    写入后 data.ctrl 直接打印: {data.ctrl}")
print(f"    （注意: MuJoCo 不会在这里立刻 clip，要等 mj_step / mj_forward）")

# ---- 3. 不 step，直接 mj_forward，看 ctrl 有没有被 clip ----
mujoco.mj_forward(model, data)
print(f"\n>>> 执行 mj_forward 后 data.ctrl: {data.ctrl}")
print(f"    → mj_forward 也不会 clip ctrl，ctrl 只是用户输入缓冲区")

# ---- 4. 再 mj_step，物理推进后 ctrl 仍会保留 10000，但 actuator_force 会被 MuJoCo 内部 clamp ----
mujoco.mj_step(model, data)
print(f"\n>>> 执行 mj_step 后 data.ctrl: {data.ctrl}")
print(f"    data.actuator_force:    {data.actuator_force}")

# ---- 5. 逐电机对比 ----
print("\n=== 逐电机对比 clamp 效果 ===")
print(f"{'ID':>3}  {'执行器':<28}{'ctrlrange':>14}{'输入ctrl':>10}{'step后ctrl':>11}{'actuator_force':>15}")
print("-" * 105)
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}"
    rng = f"[{cmin[i]:.0f},{cmax[i]:.0f}]" if clipped_flags[i] else "无"
    print(f"{i:>3}  {name:<28}{rng:>14}{INPUT_VAL:>10.0f}"
          f"{data.ctrl[i]:>11.1f}{data.actuator_force[i]:>15.1f}")

# ---- 6. 手动 np.clip 的对照 ----
print("\n=== 用 np.clip 手动 clamp（对照） ===")
clipped_by_us = np.clip(np.full(model.nu, INPUT_VAL), cmin, cmax)
print(f"手动 np.clip 结果: {clipped_by_us}")
print(f"MuJoCo actuator_force 结果: {data.actuator_force}")
match = np.allclose(clipped_by_us, data.actuator_force, rtol=1e-6, atol=1e-6)
print(f"两者是否一致: {match}  {'✓' if match else '✗'}")

# ---- 7. 负数同样验证 ----
INPUT_NEG = -9999.0
data.ctrl[:] = INPUT_NEG
mujoco.mj_step(model, data)
clipped_neg = np.clip(np.full(model.nu, INPUT_NEG), cmin, cmax)
print(f"\n=== 再验证负数输入 {INPUT_NEG} ===")
print(f"手动 np.clip 结果: {clipped_neg}")
print(f"MuJoCo actuator_force 结果: {data.actuator_force}")
match_neg = np.allclose(clipped_neg, data.actuator_force, rtol=1e-6, atol=1e-6)
print(f"两者是否一致: {match_neg}  {'✓' if match_neg else '✗'}")

print("\n结论:")
print("  - data.ctrl 是你写什么就存什么的缓冲区，MuJoCo 不会在 forward/step 里改它")
print("  - 真正的 clamp 发生在执行器内部: data.actuator_force 是 clip 之后的实际输出")
print("  - 你用 np.clip(data.ctrl, ctrl_min, ctrl_max) 是**双重保险**，让后续逻辑也安全")
