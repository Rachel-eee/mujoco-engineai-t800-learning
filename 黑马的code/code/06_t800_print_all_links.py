from pathlib import Path

import mujoco


def geom_type_name(geom_type: int) -> str:
    try:
        return mujoco.mjtGeom(geom_type).name
    except Exception:
        return str(int(geom_type))


def geom_size_desc(model: mujoco.MjModel, geom_id: int) -> str:
    gtype = geom_type_name(int(model.geom_type[geom_id]))
    gsize = model.geom_size[geom_id]
    a = float(gsize[0])
    b = float(gsize[1])
    c = float(gsize[2])

    if gtype == "BOX":
        return f"{2 * a:.3f}×{2 * b:.3f}×{2 * c:.3f}m"
    if gtype == "SPHERE":
        return f"diam={2 * a:.3f}m"
    if gtype in ("CAPSULE", "CYLINDER"):
        return f"r={a:.3f}m, L={2 * b:.3f}m"
    if gtype == "ELLIPSOID":
        return f"rx={a:.3f}m, ry={b:.3f}m, rz={c:.3f}m"

    return f"({a:.3f},{b:.3f},{c:.3f})"


def main():
    xml_path = Path(__file__).with_name("t800.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    print(f"XML文件: {xml_path.name}")
    print(f"连杆(body)数量(不含world): {model.nbody - 1}")

    rows = []
    for bid in range(1, model.nbody):
        bname = model.body(bid).name or f"<unnamed_{bid}>"
        mass = float(model.body_mass[bid])

        geom_adr = int(model.body_geomadr[bid])
        geom_num = int(model.body_geomnum[bid])

        geom_descs = []
        for gid in range(geom_adr, geom_adr + geom_num):
            gtype = geom_type_name(int(model.geom_type[gid]))
            gdesc = geom_size_desc(model, gid)
            geom_descs.append(f"{gtype}:{gdesc}")

        if not geom_descs:
            size_desc = "-"
        elif len(geom_descs) <= 2:
            size_desc = "; ".join(geom_descs)
        else:
            size_desc = f"{geom_descs[0]}; {geom_descs[1]}; +{len(geom_descs) - 2} more"

        rows.append((bid, bname, mass, size_desc))

    name_width = max(8, min(36, max(len(r[1]) for r in rows)))
    header = f"{'ID':>3}  {'LINK':<{name_width}}  {'MASS(kg)':>10}  SIZE"
    print(header)
    print("-" * len(header))

    for bid, bname, mass, size_desc in rows:
        print(f"{bid:3d}  {bname:<{name_width}}  {mass:10.4f}  {size_desc}")


if __name__ == "__main__":
    main()
