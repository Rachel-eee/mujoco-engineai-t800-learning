from pathlib import Path

import mujoco
import mujoco.viewer


def main():
    xml_path = Path(__file__).with_name("t800.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print(f"已加载: {xml_path.name}")
    print("将打开 MuJoCo 默认界面。")
    print("可在界面右侧使用 MuJoCo 自带的 actuator/motor 控制滑块。")
    print("点击运行后即可进行仿真。")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()

