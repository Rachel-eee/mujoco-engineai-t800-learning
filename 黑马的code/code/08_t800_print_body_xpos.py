from pathlib import Path

import mujoco


def main():
    xml_path = Path(__file__).with_name("t800.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    print(f"XML文件: {xml_path.name}")
    print(f"body数量(含world): {model.nbody}")

    rows = []
    for bid in range(model.nbody):
        name = model.body(bid).name or ("world" if bid == 0 else f"<unnamed_{bid}>")
        wpos = data.xpos[bid]
        rows.append((bid, name, float(wpos[0]), float(wpos[1]), float(wpos[2])))

    name_width = max(8, min(36, max(len(r[1]) for r in rows)))
    header = f"{'ID':>3}  {'BODY':<{name_width}}  {'xpos_world(x,y,z)':>22}"
    print(header)
    print("-" * len(header))

    for bid, name, wx, wy, wz in rows:
        wtxt = f"({wx: .4f},{wy: .4f},{wz: .4f})"
        print(f"{bid:3d}  {name:<{name_width}}  {wtxt:>22}")


if __name__ == "__main__":
    main()

