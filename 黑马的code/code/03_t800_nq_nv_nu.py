from pathlib import Path

import mujoco


def main():
    xml_path = Path(__file__).with_name("t800.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    print(f"XML文件: {xml_path.name}")
    print(f"nq: {model.nq}")
    print(f"nv: {model.nv}")
    print(f"nu: {model.nu}")


if __name__ == "__main__":
    main()
