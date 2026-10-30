"""用 matplotlib 3D 画出 T800 机器人的骨架（基于 body 父子关系连线）。"""

import os

import mujoco
import matplotlib.pyplot as plt
import numpy as np

# ---- 中文字体设置（Windows 下优先 SimHei / Microsoft YaHei）----
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 负号避免乱码

# ---- 加载模型并计算 xpos ----
os.chdir(os.path.dirname(os.path.abspath(__file__)))
model = mujoco.MjModel.from_xml_path("t800.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# ---- 收集父子对（跳过 world=0，本身就是根）----
edges = []
for i in range(1, model.nbody):
    p = model.body_parentid[i]
    edges.append((p, i))

# 关节分组颜色：腿 / 腰 / 臂 / 头
group_colors = {
    "leg_left": "#d62728",
    "leg_right": "#d62728",
    "body":     "#2ca02c",
    "arm_left": "#1f77b4",
    "arm_right":"#1f77b4",
    "head":     "#9467bd",
}


def edge_color(child_idx: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, child_idx) or ""
    if "_L" in name or name.startswith("LINK_FOOT_L"):
        return group_colors["leg_left"]
    if "_R" in name or name.startswith("LINK_FOOT_R"):
        return group_colors["leg_right"]
    if name.startswith("LINK_SHOULDER_PITCH_L") or name.startswith("LINK_WRIST_END_L"):
        return group_colors["arm_left"]
    if name.startswith("LINK_SHOULDER_PITCH_R") or name.startswith("LINK_WRIST_END_R"):
        return group_colors["arm_right"]
    if "HEAD" in name:
        return group_colors["head"]
    return group_colors["body"]


# ---- 3D 绘图 ----
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# 连线
for p, c in edges:
    p_pos = data.xpos[p]
    c_pos = data.xpos[c]
    color = edge_color(c)
    ax.plot(
        [p_pos[0], c_pos[0]],
        [p_pos[1], c_pos[1]],
        [p_pos[2], c_pos[2]],
        color=color, linewidth=2.0,
    )

# body 节点：普通节点小 + 黑色；关键节点大 + 彩色
all_idx = list(range(1, model.nbody))
labeled_indices = set()  # 下面会填充
regular_idx = []

# 先定义 KEY_BODIES 供后续使用
KEY_BODIES = {
    "LINK_BASE": "基座",
    "LINK_WAIST_YAW": "腰部",
    "LINK_HEAD_YAW": "头",
    "LINK_FOOT_L": "左脚",
    "LINK_FOOT_R": "右脚",
    "LINK_WRIST_END_L": "左手腕",
    "LINK_WRIST_END_R": "右手腕",
}

for i in all_idx:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
    if name in KEY_BODIES:
        labeled_indices.add(i)
    else:
        regular_idx.append(i)

# 普通节点（小+黑）
if regular_idx:
    xs_r = data.xpos[regular_idx, 0]
    ys_r = data.xpos[regular_idx, 1]
    zs_r = data.xpos[regular_idx, 2]
    ax.scatter(xs_r, ys_r, zs_r, c="black", s=20, marker="o", zorder=5)

# 关键节点（大+蓝）
if labeled_indices:
    xs_k = data.xpos[list(labeled_indices), 0]
    ys_k = data.xpos[list(labeled_indices), 1]
    zs_k = data.xpos[list(labeled_indices), 2]
    ax.scatter(xs_k, ys_k, zs_k, c="#ff7f0e", s=70, marker="*", zorder=6)

# 给关键节点加中文标签
for key_name, label in KEY_BODIES.items():
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, key_name)
    if idx >= 0:
        pos = data.xpos[idx]
        label_offset = 0.02
        ax.text(
            pos[0] + label_offset, pos[1] + label_offset, pos[2] + label_offset,
            label, fontsize=9, color="#333333",
        )

print(f"已标注 {len(labeled_indices)} 个关键节点: {sorted(labeled_indices)}")

ax.set_title("T800 机器人骨架（Matplotlib 3D）")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

# 让三轴比例一致（否则 3D 看起来会被压扁）
all_pos = data.xpos[1:]  # 跳过 world
max_range = np.array([all_pos[:, 0].max() - all_pos[:, 0].min(),
                      all_pos[:, 1].max() - all_pos[:, 1].min(),
                      all_pos[:, 2].max() - all_pos[:, 2].min()]).max() / 2.0
mid_x = (all_pos[:, 0].max() + all_pos[:, 0].min()) / 2.0
mid_y = (all_pos[:, 1].max() + all_pos[:, 1].min()) / 2.0
mid_z = (all_pos[:, 2].max() + all_pos[:, 2].min()) / 2.0
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# 固定视角：略微俯视、从斜前方看
ax.view_init(elev=25, azim=-55)

plt.tight_layout()

# 保存 + 显示，避免有些环境 show 完直接关掉
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t800_skeleton.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"骨架图已保存: {out_path}")
plt.show()
