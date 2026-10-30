from pathlib import Path

import mujoco


def main():
    xml_path = Path(__file__).with_name("t800.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    print(f"XML文件: {xml_path.name}")
    print(f"身体部件数量: {model.nbody}")
    print(f"关节数量: {model.njnt}")
    print(f"电机数量: {model.nu}")


if __name__ == "__main__":
    main()
