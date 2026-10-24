"""LeRobotDataset の特徴量定義.

mode="image": 関節角度 ＋ 天井カメラ画像（本課題の提案手法）
mode="state": 関節角度のみ（教材と同じ構成。比較用のベースライン）
"""

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "gripper"]
IMAGE_KEY = "observation.images.top"
TASK = "reach above the red cube"
ROBOT_TYPE = "crane_plus"


def make_features(mode: str, image_hw=(168, 224)) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (5,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (5,),
            "names": JOINT_NAMES,
        },
    }
    if mode == "image":
        h, w = image_hw
        features[IMAGE_KEY] = {
            # 動画(video)ではなく画像(image)で保存する。Colab での動画デコードの依存関係を避けるため。
            "dtype": "image",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    elif mode == "state":
        # ACT は「画像」か「環境状態」のどちらかが必須。教材と同じく関節角度を複製して入れる。
        features["observation.environment_state"] = {
            "dtype": "float32",
            "shape": (5,),
            "names": JOINT_NAMES,
        }
    else:
        raise ValueError(f"unknown mode: {mode}")
    return features
