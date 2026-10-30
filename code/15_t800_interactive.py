"""
15_t800_interactive.py —— 最简洁的"手动拖动关节"教学示例

运行:   python 15_t800_interactive.py
操作:   鼠标左键点击机器人任何一个 body，按住拖动
        电机 ctrl 默认全 0 → 关节在重力下自由摆动
"""

import os
import mujoco
import mujoco.viewer

# 切到脚本目录，避免 Windows 下含中文的绝对路径问题
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path("t800.xml")
data = mujoco.MjData(model)

print("=== T800 手动拖动关节演示 ===")
print("操作: 鼠标左键点击机器人任何一个 body，按住拖动")
print("      电机 ctrl 默认全 0 → 关节在重力下自由摆动")
print("      右键旋转视角 / 滚轮缩放")
print("关掉窗口即结束\n")

# launch: 交互式 viewer，鼠标可以拖 body
#   - 启动后 viewer 自己跑物理循环，你只管拖
#   - 关掉窗口后返回
mujoco.viewer.launch(model, data)

print("viewer 已关闭")
