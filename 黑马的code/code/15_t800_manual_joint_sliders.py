from pathlib import Path
import math
import tkinter as tk

import mujoco
import mujoco.viewer


def rad2deg(x: float) -> float:
    return x * 180.0 / math.pi


def deg2rad(x: float) -> float:
    return x * math.pi / 180.0


def main():
    xml_path = Path(__file__).with_name("t800.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    qpos0 = data.qpos.copy()

    joints = []
    for jid in range(model.njnt):
        jtype = int(model.jnt_type[jid])
        if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue

        name = model.joint(jid).name or f"joint_{jid}"
        qposadr = int(model.jnt_qposadr[jid])
        limited = int(model.jnt_limited[jid])
        lo = float(model.jnt_range[jid, 0])
        hi = float(model.jnt_range[jid, 1])

        if jtype == mujoco.mjtJoint.mjJNT_HINGE:
            init_value = rad2deg(float(qpos0[qposadr]))
            slider_min = rad2deg(lo) if limited else -180.0
            slider_max = rad2deg(hi) if limited else 180.0
            unit = "deg"
        else:
            init_value = float(qpos0[qposadr])
            slider_min = lo if limited else -1.0
            slider_max = hi if limited else 1.0
            unit = "m"

        joints.append(
            {
                "jid": jid,
                "name": name,
                "type": jtype,
                "qposadr": qposadr,
                "init_value": init_value,
                "slider_min": slider_min,
                "slider_max": slider_max,
                "unit": unit,
            }
        )

    root = tk.Tk()
    root.title("T800 手动关节控制")
    root.geometry("900x800")

    top_label = tk.Label(
        root,
        text="拖动滑条，直接修改各关节 qpos。这个版本是教学演示版，不走 PD，不走动力学控制。",
        anchor="w",
        justify="left",
    )
    top_label.pack(fill="x", padx=10, pady=8)

    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)
    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    slider_vars = []
    for i, info in enumerate(joints):
        row = tk.Frame(scroll_frame)
        row.pack(fill="x", padx=10, pady=4)

        name_label = tk.Label(row, text=f"[{i:02d}] {info['name']}", width=28, anchor="w")
        name_label.pack(side="left")

        value_var = tk.DoubleVar(value=info["init_value"])
        slider_vars.append(value_var)

        value_text = tk.StringVar(value=f"{info['init_value']:.2f} {info['unit']}")
        value_label = tk.Label(row, textvariable=value_text, width=12, anchor="e")
        value_label.pack(side="right")

        def on_change(v, text_var=value_text, unit=info["unit"]):
            text_var.set(f"{float(v):.2f} {unit}")

        scale = tk.Scale(
            row,
            from_=info["slider_min"],
            to=info["slider_max"],
            resolution=0.1 if info["unit"] == "deg" else 0.001,
            orient="horizontal",
            variable=value_var,
            command=on_change,
            length=500,
        )
        scale.pack(side="right", padx=8)

    btn_row = tk.Frame(root)
    btn_row.pack(fill="x", padx=10, pady=8)

    def reset_all():
        for info, value_var in zip(joints, slider_vars):
            value_var.set(info["init_value"])

    tk.Button(btn_row, text="恢复初始姿态", command=reset_all).pack(side="left")

    viewer = mujoco.viewer.launch_passive(model, data)

    def update():
        if not viewer.is_running():
            root.destroy()
            return

        for info, value_var in zip(joints, slider_vars):
            qposadr = info["qposadr"]
            value = float(value_var.get())
            if info["type"] == mujoco.mjtJoint.mjJNT_HINGE:
                data.qpos[qposadr] = deg2rad(value)
            else:
                data.qpos[qposadr] = value

        data.qvel[:] = 0
        if model.nu > 0:
            data.ctrl[:] = 0
        mujoco.mj_forward(model, data)
        viewer.sync()
        root.after(30, update)

    def on_close():
        try:
            viewer.close()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    print(f"已加载: {xml_path.name}")
    print(f"可手动拖动的关节数量: {len(joints)}")
    update()
    root.mainloop()


if __name__ == "__main__":
    main()

