from pathlib import Path

import mujoco


def main():
    xml_path = Path(__file__).with_name("t800.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    print(f"XML文件: {xml_path.name}")
    print(f"body数量(含world): {model.nbody}")

    rows = []
    for bid in range(model.nbody):
        name = model.body(bid).name or ("world" if bid == 0 else f"<unnamed_{bid}>")
        lpos = model.body_pos[bid]
        rows.append((bid, name, float(lpos[0]), float(lpos[1]), float(lpos[2])))

    name_width = max(8, min(36, max(len(r[1]) for r in rows)))
    header = f"{'ID':>3}  {'BODY':<{name_width}}  {'body_pos(x,y,z)':>22}"
    print(header)
    print("-" * len(header))

    for bid, name, lx, ly, lz in rows:
        ltxt = f"({lx: .4f},{ly: .4f},{lz: .4f})"
        print(f"{bid:3d}  {name:<{name_width}}  {ltxt:>22}")


if __name__ == "__main__":
    main()

