"""
16_t800_mujoco_gui.py —— 纯 MuJoCo 默认 GUI（无任何外部库）
- 使用 mujoco.viewer.launch（交互式模式）
- viewer 自带电机 ctrl 滑块面板：按 H 切换显示
- 滑块直接改 data.ctrl，MuJoCo 内部会按 ctrlrange 自动 clamp
"""

import os
import mujoco
import mujoco.viewer

# 切到脚本目录，避免 Windows 下含中文的绝对路径问题
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path("t800.xml")
data = mujoco.MjData(model)

print("=" * 60)
print("  T800 —— MuJoCo 自带 GUI 控制滑块演示")
print("=" * 60)
print("  操作说明:")
print("    H        切换 ctrl 滑块面板显示")
print("    SPC      暂停 / 继续")
print("    =        单步执行 1 帧 (暂停时有效)")
print("    .        连续单步")
print("    R        重置仿真")
print("    左键拖拽 body   手动拖动")
print("    右键拖拽         旋转视角")
print("    滚轮            缩放")
print("=" * 60)
print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}  (电机数={model.nu})")
print("  滑块范围来自 XML 的 ctrlrange，MuJoCo 内部自动 clamp")
print("  关掉窗口即结束\n")

# launch: MuJoCo 原生交互式 viewer
#   - 默认就有菜单栏，菜单栏 → Control 会显示所有电机的 ctrl 滑块
#   - 快捷键 H 也能直接弹出/收起滑块面板
#   - 滑块值会实时写到 data.ctrl，mj_step 里 MuJoCo 内部会按 ctrlrange 自动 clip
mujoco.viewer.launch(model, data)

print("viewer 已关闭")
