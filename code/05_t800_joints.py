"""打印 T800 机器人的所有关节信息（ID / 名称 / 类型 / qpos 与 dof 地址）。"""

import os
import mujoco

# 先切到脚本目录，避免 Windows 下含中文的绝对路径无法被 MuJoCo 打开
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = mujoco.MjModel.from_xml_path("t800.xml")

JOINT_TYPES = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
}

print(f"=== T800 关节列表（共 {model.njnt} 个） ===")
print(f"{'ID':>3}  {'名称':<22}{'类型':<7}{'qpos_adr':>9}{'dof_adr':>8}")
print("-" * 54)

for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "(未命名)"
    jtype = JOINT_TYPES.get(model.jnt_type[i], "unknown")
    print(f"{i:>3}  {name:<22}{jtype:<7}{model.jnt_qposadr[i]:>9}{model.jnt_dofadr[i]:>8}")
