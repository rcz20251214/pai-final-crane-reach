#!/usr/bin/env python3
"""学習した ACT 方策を Gazebo 上の CRANE+ V2 で動かす推論ノード.

教材の pai_training/scripts/infer.py をもとに、
  - 天井カメラ画像（/image_raw）を入力に追加（方策が画像を使う場合のみ）
  - シミュレーション時間で 10Hz（学習データと同じ周期）で動かす
  - /policy/start, /policy/stop サービスで開始・停止できる（評価ノードから使う）

使い方（ROS を source してから pai_training の pixi 環境で実行）:
  source ~/pai_ws/install/setup.bash
  cd ~/pai_training
  pixi run python ~/pai_ws/src/pai-final-crane-reach/training/infer_node.py \
      --model-path ~/checkpoints/act_image/last
"""

import argparse
import threading

import numpy as np
import rclpy
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM = ["crane_plus_joint1", "crane_plus_joint2", "crane_plus_joint3", "crane_plus_joint4"]
HAND = "crane_plus_joint_hand"
IMAGE_KEY = "observation.images.top"


class PolicyNode(Node):
    def __init__(self, model_path, rate_hz, n_action_steps, autostart, device, temporal_ensemble=None):
        super().__init__("act_policy",
                         parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.device = torch.device(device)
        cfg = PreTrainedConfig.from_pretrained(model_path)
        if temporal_ensemble is not None:
            # 時間的アンサンブル: 毎ステップ推論し、重なった予測を指数重みで平均する（ACT論文の手法）
            cfg.temporal_ensemble_coeff = temporal_ensemble
            cfg.n_action_steps = 1
        elif n_action_steps is not None:
            cfg.n_action_steps = n_action_steps
        self.policy = ACTPolicy.from_pretrained(model_path, config=cfg).to(self.device)
        self.policy.eval()
        self.input_keys = list(self.policy.config.input_features.keys())
        self.use_image = IMAGE_KEY in self.input_keys
        self.get_logger().info(
            f"入力: {self.input_keys} / n_action_steps={self.policy.config.n_action_steps}"
            f" / temporal_ensemble={self.policy.config.temporal_ensemble_coeff}")

        self.arm_pub = self.create_publisher(JointTrajectory, "/crane_plus_arm_controller/joint_trajectory", 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory, "/crane_plus_gripper_controller/joint_trajectory", 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint, 10)
        if self.use_image:
            self.create_subscription(Image, "/image_raw", self._on_image, 10)
        self.create_service(Trigger, "/policy/start", self._srv_start)
        self.create_service(Trigger, "/policy/stop", self._srv_stop)

        self._lock = threading.Lock()
        self.state = None
        self.image = None
        self.active = autostart
        self.period = 1.0 / rate_hz
        self.create_timer(self.period, self._step)

    def _on_joint(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(n in pos for n in ARM + [HAND]):
            with self._lock:
                self.state = np.array([pos[n] for n in ARM + [HAND]], dtype=np.float32)

    def _on_image(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        img = img[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1]
        with self._lock:
            self.image = img.copy()

    def _srv_start(self, req, res):
        with self._lock:
            self.policy.reset()  # 行動キューを空にする
            self.active = True
        res.success, res.message = True, "started"
        return res

    def _srv_stop(self, req, res):
        with self._lock:
            self.active = False
        res.success, res.message = True, "stopped"
        return res

    def _step(self):
        with self._lock:
            if not self.active or self.state is None or (self.use_image and self.image is None):
                return
            state, image = self.state.copy(), None if self.image is None else self.image.copy()

        obs = {"observation.state": torch.from_numpy(state).unsqueeze(0)}
        if "observation.environment_state" in self.input_keys:
            obs["observation.environment_state"] = obs["observation.state"].clone()
        if self.use_image:
            # データセットと同じく (1, 3, H, W)、0〜1 の float にする
            obs[IMAGE_KEY] = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        obs = {k: v.to(self.device) for k, v in obs.items()}

        with torch.no_grad():
            action = self.policy.select_action(obs).squeeze(0).cpu().numpy()

        jt = JointTrajectory()
        jt.joint_names = ARM
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in action[:4]]
        pt.time_from_start.nanosec = int(self.period * 1e9)
        jt.points.append(pt)
        self.arm_pub.publish(jt)

        jg = JointTrajectory()
        jg.joint_names = [HAND]
        pg = JointTrajectoryPoint()
        pg.positions = [float(action[4])]
        pg.time_from_start.nanosec = int(self.period * 1e9)
        jg.points.append(pg)
        self.gripper_pub.publish(jg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--rate", type=float, default=10.0, help="制御周期 [Hz]（学習データの fps と合わせる）")
    ap.add_argument("--n-action-steps", type=int, default=None, help="指定すると学習時の値を上書き")
    ap.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF",
                    help="時間的アンサンブルを使う（例: 0.01）。指定すると n_action_steps は 1 になる")
    ap.add_argument("--no-autostart", action="store_true", help="/policy/start が呼ばれるまで動かさない")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = PolicyNode(args.model_path, args.rate, args.n_action_steps,
                      not args.no_autostart, args.device, args.temporal_ensemble)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
