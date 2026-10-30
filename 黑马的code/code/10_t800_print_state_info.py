from pathlib import Path

import mujoco


def format_array(arr) -> str:
    return "[" + ", ".join(f"{float(x): .6f}" for x in arr) + "]"


def main():
    xml_path = Path(__file__).with_name("t800.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print(f"XML文件: {xml_path.name}")
    print("-" * 60)
    print(f"nq   数量: {model.nq}")
    print(f"nv   数量: {model.nv}")
    print(f"nu   数量: {model.nu}")
    print("-" * 60)
    print(f"qpos 长度: {len(data.qpos)}")
    print(f"qpos 初始值: {format_array(data.qpos)}")
    print("-" * 60)
    print(f"qvel 长度: {len(data.qvel)}")
    print(f"qvel 初始值: {format_array(data.qvel)}")
    print("-" * 60)
    print(f"ctrl 长度: {len(data.ctrl)}")
    print(f"ctrl 初始值: {format_array(data.ctrl)}")


if __name__ == "__main__":
    main()

