from pathlib import Path

import mujoco


def joint_type_name(jnt_type: int) -> str:
    try:
        return mujoco.mjtJoint(jnt_type).name
    except Exception:
        return str(int(jnt_type))


def main():
    xml_path = Path(__file__).with_name("t800.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    print(f"XML文件: {xml_path.name}")
    print(f"关节数量: {model.njnt}")

    for jid in range(model.njnt):
        name = model.joint(jid).name or f"<unnamed_{jid}>"
        jtype = joint_type_name(int(model.jnt_type[jid]))
        qposadr = int(model.jnt_qposadr[jid])
        dofadr = int(model.jnt_dofadr[jid])
        print(f"[{jid:03d}] {name} | type={jtype} | qposadr={qposadr} | dofadr={dofadr}")


if __name__ == "__main__":
    main()

