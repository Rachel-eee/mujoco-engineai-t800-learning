from pathlib import Path

import mujoco


def main():
    xml_path = Path(__file__).with_name("t800.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    print(f"XML文件: {xml_path.name}")
    print(f"电机数量(nu): {model.nu}")

    rows = []
    for aid in range(model.nu):
        aname = model.actuator(aid).name or f"<unnamed_actuator_{aid}>"
        jnt_id = int(model.actuator_trnid[aid, 0])
        jname = model.joint(jnt_id).name if 0 <= jnt_id < model.njnt else "<none>"

        limited = int(model.actuator_ctrllimited[aid])
        lo, hi = float(model.actuator_ctrlrange[aid, 0]), float(model.actuator_ctrlrange[aid, 1])
        rows.append((aid, aname, jname, limited, lo, hi))

    a_width = max(8, min(32, max(len(r[1]) for r in rows))) if rows else 8
    j_width = max(8, min(32, max(len(r[2]) for r in rows))) if rows else 8

    header = f"{'ID':>3}  {'ACTUATOR':<{a_width}}  {'JOINT':<{j_width}}  {'LIMIT':>5}  {'CTRL_LO':>10}  {'CTRL_HI':>10}"
    print(header)
    print("-" * len(header))

    for aid, aname, jname, limited, lo, hi in rows:
        limit_txt = "yes" if limited else "no"
        lo_txt = f"{lo:.6f}" if limited else "-"
        hi_txt = f"{hi:.6f}" if limited else "-"
        print(f"{aid:3d}  {aname:<{a_width}}  {jname:<{j_width}}  {limit_txt:>5}  {lo_txt:>10}  {hi_txt:>10}")


if __name__ == "__main__":
    main()

