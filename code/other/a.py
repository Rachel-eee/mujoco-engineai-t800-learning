import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.font_manager as fm

# ===== 解决中文乱码 =====
# 方法1: 使用系统自带中文字体（推荐）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 如果上面不行，可以用方法2: 指定字体文件路径
# font_path = '/System/Library/Fonts/PingFang.ttc'  # Mac
# font_path = 'C:/Windows/Fonts/msyh.ttc'  # Windows
# prop = fm.FontProperties(fname=font_path)
# 然后在每个text/title中加 fontproperties=prop

# ===== 所有关节点坐标 =====
joints = {
    "BASE": [0, 0, 1.03],
    "HIP_PITCH_L": [0, 0.079178, 1.03],
    "HIP_ROLL_L": [0.004662, 0.1195, 0.92248],
    "HIP_YAW_L": [0.0141236, 0.1195, 0.841968],
    "KNEE_PITCH_L": [-0.0235834, 0.1195, 0.554048],
    "ANKLE_PITCH_L": [-0.0263002, 0.1195, 0.115378],
    "ANKLE_ROLL_L": [-0.0251929, 0.1195, 0.07745],
    "FOOT_L": [-0.0251929, 0.1195, 0.07745],
    "HIP_PITCH_R": [0, -0.079178, 1.03],
    "HIP_ROLL_R": [0.004662, -0.1195, 0.92248],
    "HIP_YAW_R": [0.0141236, -0.1195, 0.841968],
    "KNEE_PITCH_R": [-0.0235834, -0.1195, 0.554048],
    "ANKLE_PITCH_R": [-0.0263002, -0.1195, 0.115378],
    "ANKLE_ROLL_R": [-0.0251929, -0.1195, 0.07745],
    "FOOT_R": [-0.0251929, -0.1195, 0.07745],
    "WAIST_YAW": [0, 0, 1.13384],
    "SHOULDER_PITCH_L": [-0.011, 0.16303, 1.41273],
    "SHOULDER_ROLL_L": [-0.011, 0.225563, 1.397275],
    "SHOULDER_YAW_L": [-0.0109654, 0.2340876, 1.314563],
    "ELBOW_PITCH_L": [-0.0090402, 0.2476846, 1.163813],
    "ELBOW_YAW_L": [-0.0112493, 0.2571208, 1.031993],
    "WRIST_END_L": [0.0181867, 0.2696063, 0.89977149],
    "SHOULDER_PITCH_R": [-0.011, -0.16303, 1.41273],
    "SHOULDER_ROLL_R": [-0.011, -0.225563, 1.397275],
    "SHOULDER_YAW_R": [-0.0109654, -0.2340876, 1.314563],
    "ELBOW_PITCH_R": [-0.0090402, -0.2476846, 1.163813],
    "ELBOW_YAW_R": [-0.0112493, -0.2571208, 1.031993],
    "WRIST_END_R": [0.0181867, -0.2696063, 0.89977149],
    "HEAD_PITCH": [-0.0113, 0, 1.51988],
    "HEAD_YAW": [-0.0113, 0, 1.560079],
}

# ===== 关节中文翻译 =====
joints_cn = {
    "BASE": "基座",
    "HIP_PITCH_L": "左髋俯仰",
    "HIP_ROLL_L": "左髋翻滚",
    "HIP_YAW_L": "左髋偏航",
    "KNEE_PITCH_L": "左膝",
    "ANKLE_PITCH_L": "左踝俯仰",
    "ANKLE_ROLL_L": "左踝翻滚",
    "FOOT_L": "左脚",
    "HIP_PITCH_R": "右髋俯仰",
    "HIP_ROLL_R": "右髋翻滚",
    "HIP_YAW_R": "右髋偏航",
    "KNEE_PITCH_R": "右膝",
    "ANKLE_PITCH_R": "右踝俯仰",
    "ANKLE_ROLL_R": "右踝翻滚",
    "FOOT_R": "右脚",
    "WAIST_YAW": "腰部",
    "SHOULDER_PITCH_L": "左肩俯仰",
    "SHOULDER_ROLL_L": "左肩翻滚",
    "SHOULDER_YAW_L": "左肩偏航",
    "ELBOW_PITCH_L": "左肘",
    "ELBOW_YAW_L": "左肘偏航",
    "WRIST_END_L": "左手腕",
    "SHOULDER_PITCH_R": "右肩俯仰",
    "SHOULDER_ROLL_R": "右肩翻滚",
    "SHOULDER_YAW_R": "右肩偏航",
    "ELBOW_PITCH_R": "右肘",
    "ELBOW_YAW_R": "右肘偏航",
    "WRIST_END_R": "右手腕",
    "HEAD_PITCH": "头部俯仰",
    "HEAD_YAW": "头部",
}

# ===== 身体部位分组（用于颜色标注）=====
body_parts = {
    "躯干": ["BASE", "WAIST_YAW"],
    "左腿": ["HIP_PITCH_L", "HIP_ROLL_L", "HIP_YAW_L", "KNEE_PITCH_L", "ANKLE_PITCH_L", "ANKLE_ROLL_L", "FOOT_L"],
    "右腿": ["HIP_PITCH_R", "HIP_ROLL_R", "HIP_YAW_R", "KNEE_PITCH_R", "ANKLE_PITCH_R", "ANKLE_ROLL_R", "FOOT_R"],
    "左臂": ["SHOULDER_PITCH_L", "SHOULDER_ROLL_L", "SHOULDER_YAW_L", "ELBOW_PITCH_L", "ELBOW_YAW_L", "WRIST_END_L"],
    "右臂": ["SHOULDER_PITCH_R", "SHOULDER_ROLL_R", "SHOULDER_YAW_R", "ELBOW_PITCH_R", "ELBOW_YAW_R", "WRIST_END_R"],
    "头部": ["HEAD_PITCH", "HEAD_YAW"],
}

# 颜色映射
part_colors = {
    "躯干": "green",
    "左腿": "red",
    "右腿": "blue",
    "左臂": "orange",
    "右臂": "purple",
    "头部": "pink",
}

# ===== 定义骨架连接 =====
connections = [
    ("BASE", "WAIST_YAW"),
    ("WAIST_YAW", "HEAD_PITCH"),
    ("HEAD_PITCH", "HEAD_YAW"),
    ("BASE", "HIP_PITCH_L"),
    ("HIP_PITCH_L", "HIP_ROLL_L"),
    ("HIP_ROLL_L", "HIP_YAW_L"),
    ("HIP_YAW_L", "KNEE_PITCH_L"),
    ("KNEE_PITCH_L", "ANKLE_PITCH_L"),
    ("ANKLE_PITCH_L", "ANKLE_ROLL_L"),
    ("ANKLE_ROLL_L", "FOOT_L"),
    ("BASE", "HIP_PITCH_R"),
    ("HIP_PITCH_R", "HIP_ROLL_R"),
    ("HIP_ROLL_R", "HIP_YAW_R"),
    ("HIP_YAW_R", "KNEE_PITCH_R"),
    ("KNEE_PITCH_R", "ANKLE_PITCH_R"),
    ("ANKLE_PITCH_R", "ANKLE_ROLL_R"),
    ("ANKLE_ROLL_R", "FOOT_R"),
    ("WAIST_YAW", "SHOULDER_PITCH_L"),
    ("SHOULDER_PITCH_L", "SHOULDER_ROLL_L"),
    ("SHOULDER_ROLL_L", "SHOULDER_YAW_L"),
    ("SHOULDER_YAW_L", "ELBOW_PITCH_L"),
    ("ELBOW_PITCH_L", "ELBOW_YAW_L"),
    ("ELBOW_YAW_L", "WRIST_END_L"),
    ("WAIST_YAW", "SHOULDER_PITCH_R"),
    ("SHOULDER_PITCH_R", "SHOULDER_ROLL_R"),
    ("SHOULDER_ROLL_R", "SHOULDER_YAW_R"),
    ("SHOULDER_YAW_R", "ELBOW_PITCH_R"),
    ("ELBOW_PITCH_R", "ELBOW_YAW_R"),
    ("ELBOW_YAW_R", "WRIST_END_R"),
]

def get_pos(name):
    return np.array(joints[name])

# ===== 创建图形 =====
fig = plt.figure(figsize=(18, 7))

# ===== 子图1: 3D立体视图（带颜色分组）=====
ax1 = fig.add_subplot(131, projection='3d')
ax1.set_title("🤖 机器人3D骨架示意图", fontsize=14, fontweight='bold')

# 绘制每个身体部分
for part_name, joint_list in body_parts.items():
    color = part_colors[part_name]
    # 画关节点
    for name in joint_list:
        pos = joints[name]
        ax1.scatter(pos[0], pos[1], pos[2], c=color, s=100, zorder=5, 
                   edgecolors='black', linewidth=1)
        # 显示中文标签（简化显示）
        label = joints_cn.get(name, name)
        ax1.text(pos[0], pos[1], pos[2]+0.03, label, fontsize=7, ha='center',
                color='black', weight='bold')

# 画连接线（按颜色分组）
for start, end in connections:
    p1 = get_pos(start)
    p2 = get_pos(end)
    # 判断属于哪个部分
    color = 'gray'
    for part_name, joint_list in body_parts.items():
        if start in joint_list and end in joint_list:
            color = part_colors[part_name]
            break
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
             color=color, linewidth=3, alpha=0.8)

ax1.set_xlabel("X (前后)", fontsize=10)
ax1.set_ylabel("Y (左右)", fontsize=10)
ax1.set_zlabel("Z (高度)", fontsize=10)
ax1.view_init(elev=25, azim=-70)
ax1.set_box_aspect([1, 0.6, 1])

# 图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=part_colors[p], label=p, alpha=0.7) 
                   for p in part_colors]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=8)

# ===== 子图2: 正面视图 =====
ax2 = fig.add_subplot(132)
ax2.set_title("正面视图 (X-Z)  左右对称", fontsize=12, fontweight='bold')

for part_name, joint_list in body_parts.items():
    color = part_colors[part_name]
    for name in joint_list:
        pos = joints[name]
        ax2.scatter(pos[0], pos[2], c=color, s=50, zorder=5, edgecolors='black', linewidth=1)
        # 只标注关键关节
        if any(key in name for key in ["BASE", "HEAD", "HIP", "KNEE", "ANKLE", "SHOULDER", "ELBOW", "WRIST"]):
            label = joints_cn.get(name, name)
            ax2.text(pos[0], pos[2]+0.015, label, fontsize=6, ha='center')

for start, end in connections:
    p1 = get_pos(start)
    p2 = get_pos(end)
    color = 'gray'
    for part_name, joint_list in body_parts.items():
        if start in joint_list and end in joint_list:
            color = part_colors[part_name]
            break
    ax2.plot([p1[0], p2[0]], [p1[2], p2[2]], color=color, linewidth=2, alpha=0.6)

ax2.set_xlabel("X (前后)", fontsize=10)
ax2.set_ylabel("Z (高度)", fontsize=10)
ax2.axhline(y=1.03, color='gray', linestyle='--', alpha=0.3)
ax2.grid(True, alpha=0.2)
ax2.set_aspect('equal')

# ===== 子图3: 侧面视图 =====
ax3 = fig.add_subplot(133)
ax3.set_title("侧面视图 (Y-Z)  左右分布", fontsize=12, fontweight='bold')

for part_name, joint_list in body_parts.items():
    color = part_colors[part_name]
    for name in joint_list:
        pos = joints[name]
        ax3.scatter(pos[1], pos[2], c=color, s=50, zorder=5, edgecolors='black', linewidth=1)
        if any(key in name for key in ["BASE", "HEAD", "HIP", "KNEE", "ANKLE", "SHOULDER", "ELBOW", "WRIST"]):
            label = joints_cn.get(name, name)
            ax3.text(pos[1], pos[2]+0.015, label, fontsize=6, ha='center')

for start, end in connections:
    p1 = get_pos(start)
    p2 = get_pos(end)
    color = 'gray'
    for part_name, joint_list in body_parts.items():
        if start in joint_list and end in joint_list:
            color = part_colors[part_name]
            break
    ax3.plot([p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=2, alpha=0.6)

ax3.set_xlabel("Y (左右)", fontsize=10)
ax3.set_ylabel("Z (高度)", fontsize=10)
ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
ax3.grid(True, alpha=0.2)
ax3.set_aspect('equal')

plt.tight_layout()
plt.savefig('robot_skeleton_cn.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== 打印关键信息 =====
print("\n" + "="*60)
print("📐 机器人关键尺寸信息")
print("="*60)
print(f"🤖 总高度: {joints['HEAD_YAW'][2]:.3f} m")
print(f"📏 躯干高度: {joints['WAIST_YAW'][2] - joints['BASE'][2]:.3f} m")
print(f"🦵 左腿长度: {joints['BASE'][2] - joints['FOOT_L'][2]:.3f} m")
print(f"🦵 右腿长度: {joints['BASE'][2] - joints['FOOT_R'][2]:.3f} m")
print(f"💪 左臂长度: {joints['WAIST_YAW'][2] - joints['WRIST_END_L'][2]:.3f} m")
print(f"💪 右臂长度: {joints['WAIST_YAW'][2] - joints['WRIST_END_R'][2]:.3f} m")
print(f"↔️  左右腿间距: {abs(joints['HIP_PITCH_L'][1] - joints['HIP_PITCH_R'][1]):.4f} m")
print(f"↔️  左右肩间距: {abs(joints['SHOULDER_PITCH_L'][1] - joints['SHOULDER_PITCH_R'][1]):.4f} m")

print("\n" + "="*60)
print("📝 关节数量统计")
print("="*60)
print(f"🟢 躯干: {len(body_parts['躯干'])} 个关节")
print(f"🔴 左腿: {len(body_parts['左腿'])} 个关节")
print(f"🔵 右腿: {len(body_parts['右腿'])} 个关节")
print(f"🟠 左臂: {len(body_parts['左臂'])} 个关节")
print(f"🟣 右臂: {len(body_parts['右臂'])} 个关节")
print(f"🩷 头部: {len(body_parts['头部'])} 个关节")
print(f"📊 总计: {sum(len(v) for v in body_parts.values())} 个关节")