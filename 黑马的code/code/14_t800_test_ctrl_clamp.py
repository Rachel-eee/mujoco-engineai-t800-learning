from pathlib import Path

import mujoco


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def main():
    xml_path = Path(__file__).with_name("t800.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    raw_u = 10000.0

    print(f"XML文件: {xml_path.name}")
    print(f"测试输入控制值: {raw_u}")
    print(f"电机数量(nu): {model.nu}")

    rows = []
    for aid in range(model.nu):
        aname = model.actuator(aid).name or f"<unnamed_actuator_{aid}>"
        limited = int(model.actuator_ctrllimited[aid])

        if limited:
            lo = float(model.actuator_ctrlrange[aid, 0])
            hi = float(model.actuator_ctrlrange[aid, 1])
            real_u = clamp(raw_u, lo, hi)
            range_txt = f"[{lo:.6f}, {hi:.6f}]"
        else:
            real_u = raw_u
            range_txt = "-"

        data.ctrl[aid] = real_u
        rows.append((aid, aname, limited, range_txt, raw_u, float(data.ctrl[aid])))

    name_width = max(8, min(32, max(len(r[1]) for r in rows))) if rows else 8
    header = (
        f"{'ID':>3}  {'ACTUATOR':<{name_width}}  {'LIMIT':>5}  "
        f"{'CTRLRANGE':>22}  {'INPUT':>12}  {'REAL_CTRL':>12}"
    )
    print(header)
    print("-" * len(header))

    for aid, aname, limited, range_txt, input_u, real_u in rows:
        limit_txt = "yes" if limited else "no"
        print(
            f"{aid:3d}  {aname:<{name_width}}  {limit_txt:>5}  "
            f"{range_txt:>22}  {input_u:12.4f}  {real_u:12.4f}"
        )


if __name__ == "__main__":
    main()

