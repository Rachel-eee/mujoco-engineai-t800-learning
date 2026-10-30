from pathlib import Path

import mujoco


def main():
    label_mode = "leaf"
    key_name_patterns = ("FOOT", "WRIST", "HEAD", "BASE", "WAIST")

    xml_path = Path(__file__).with_name("t800.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False

    xs = data.xpos[:, 0].copy()
    ys = data.xpos[:, 1].copy()
    zs = data.xpos[:, 2].copy()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(xs, ys, zs, s=12, c="#1f77b4")

    children_count = [0] * int(model.nbody)
    for bid in range(1, model.nbody):
        children_count[int(model.body_parentid[bid])] += 1

    label_body_ids = []
    if label_mode == "leaf":
        label_body_ids = [bid for bid in range(1, model.nbody) if children_count[bid] == 0]
    elif label_mode == "key":
        for bid in range(1, model.nbody):
            name = model.body(bid).name or ""
            name_u = name.upper()
            if children_count[bid] == 0 or any(p in name_u for p in key_name_patterns):
                label_body_ids.append(bid)

    for bid in label_body_ids:
        name = model.body(bid).name or f"<unnamed_{bid}>"
        ax.text(float(xs[bid]), float(ys[bid]), float(zs[bid]), name, fontsize=7, color="#111111")

    for bid in range(1, model.nbody):
        pid = int(model.body_parentid[bid])
        ax.plot(
            [xs[pid], xs[bid]],
            [ys[pid], ys[bid]],
            [zs[pid], zs[bid]],
            color="#333333",
            linewidth=1.2,
        )

    ax.set_title(f"T800 机器人骨架结构（body xpos, label={label_mode}）")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    z_min, z_max = float(zs.min()), float(zs.max())

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cz = 0.5 * (z_min + z_max)
    r = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.55
    if r <= 0:
        r = 1.0

    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
