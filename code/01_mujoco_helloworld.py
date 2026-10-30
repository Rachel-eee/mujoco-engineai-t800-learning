"""
MuJoCo 木块自由落体 HelloWorld
- 内嵌 MJCF XML: 地面 + freejoint 木块
- 初始质心高度: 1 m
- 每隔 200 ms 打印一次木块高度（质心 z 坐标）
"""

import mujoco

# ---------- 1. 用 MJCF XML 定义世界 ----------
# 世界坐标系: z 轴向上，地面在 z=0 平面
MJCF_XML = """
<mujoco model="freefall">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <worldbody>
    <!-- 地面 -->
    <geom name="ground" type="plane" size="0 0 0.05" rgba="0.8 0.8 0.8 1"/>

    <!-- 木块: size = 半尺寸 (x, y, z) = 0.1m 的立方体 -->
    <body name="block" pos="0 0 1.0">
      <freejoint name="block_joint"/>
      <geom name="block_geom" type="box" size="0.1 0.1 0.1"
            rgba="0.2 0.6 1.0 1" mass="1.0" condim="3"/>
    </body>
  </worldbody>
</mujoco>
"""

# ---------- 2. 加载模型 ----------
model = mujoco.MjModel.from_xml_string(MJCF_XML)
data = mujoco.MjData(model)

# ---------- 3. 仿真参数 ----------
# MuJoCo 默认 dt=0.002s, 200ms = 0.2s
PRINT_INTERVAL = 0.2
dt = model.opt.timestep                      # 0.002
interval_steps = round(PRINT_INTERVAL / dt)  # 100 步

# 木块半高度 (box size z=0.1)
BLOCK_HALF_Z = 0.1

print("=== MuJoCo 木块自由落体 ===")
print(f"仿真步长 dt = {dt} s, 每 {interval_steps} 步打印一次 ({PRINT_INTERVAL*1000:.0f} ms)")
print(f"{'时间(s)':>10}  {'质心高度(m)':>12}  {'底面高度(m)':>12}")
print("-" * 40)

# 落地阈值: 底面接触地面 (质心高度 ≈ BLOCK_HALF_Z)
GROUND_LEVEL = BLOCK_HALF_Z

MAX_STEPS = 5000  # 最多跑 10s，防止落地后一直跑

for step in range(MAX_STEPS):
    mujoco.mj_step(model, data)

    # 每隔 interval_steps 步打印一次
    if step % interval_steps == 0:
        com_z = data.qpos[2]                        # 质心 z
        bottom_z = com_z - BLOCK_HALF_Z            # 底面离地高度
        print(f"{data.time:>10.3f}  {com_z:>12.4f}  {bottom_z:>12.4f}")

    # 落地后多跑几步让它稳定，然后停止
    if step > 0 and data.qpos[2] <= GROUND_LEVEL + 0.005:
        # 再多走几个 step 让接触稳定后再打印一次最终值
        for _ in range(5):
            mujoco.mj_step(model, data)
        com_z = data.qpos[2]
        bottom_z = com_z - BLOCK_HALF_Z
        print("-" * 40)
        print(f"[落地]  t={data.time:.3f}s  质心高度={com_z:.4f}m  底面高度={bottom_z:.4f}m")
        break
else:
    print("\n[超时] 仿真到达最大步数。")
