"""お手本（エキスパート）動作を自動生成し、1エピソード=1 rosbag で記録するノード.

1エピソードの流れ:
  1. アームを初期姿勢へ戻す（記録しない）
  2. 赤い立方体をランダムな位置に置き、カメラ画像が更新されるのを待つ
  3. rosbag の記録を開始
  4. 逆運動学で「立方体の上空 → 立方体の真上」へ手を伸ばす動作を生成して実行
  5. 記録を停止し、結果（手先と目標の距離など）を episodes.jsonl に追記

記録するトピック:
  /joint_states, /image_raw            … 観測（ヘッダの時刻はシミュレーション時間）
  /demo_action                          … 指令した関節角（ヘッダにシミュレーション時刻）
  /crane_plus_*_controller/joint_trajectory … 教材の変換スクリプト互換用
  /target_cube_pose                     … 立方体の位置（評価用。学習の入力には使わない）

使い方:
  ros2 run pai_reach_sim scripted_demo --ros-args \
      -p num_episodes:=50 -p out_dir:=/home/ubuntu/dataset/rosbags/reach
"""

import json
import math
import os
import random
import signal
import subprocess
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from pai_reach_sim import cube
from pai_reach_sim import kinematics as K

CONTROL_HZ = 20.0
HOME = np.zeros(4)

RECORD_TOPICS = [
    '/joint_states',
    '/image_raw',
    '/demo_action',
    '/target_cube_pose',
    '/crane_plus_arm_controller/joint_trajectory',
    '/crane_plus_gripper_controller/joint_trajectory',
]


class ScriptedDemo(Node):

    def __init__(self, node_name='scripted_demo'):
        super().__init__(
            node_name,
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.declare_parameter('num_episodes', 10)
        self.declare_parameter('start_index', 0)
        self.declare_parameter('out_dir', os.path.expanduser('~/dataset/rosbags/reach'))
        self.declare_parameter('seed', 0)
        self.declare_parameter('hover', 0.02)        # 立方体上面から手先までの高さ [m]
        self.declare_parameter('approach', 0.06)     # 上空で一度止まる高さ（hover に加算）[m]
        self.declare_parameter('home_noise', 0.10)   # 初期姿勢のばらつき [rad]
        self.declare_parameter('final_hold', 1.0)    # 目標姿勢で止まって記録し続ける時間 [s]

        self.arm_pub = self.create_publisher(
            JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        self.action_pub = self.create_publisher(JointState, '/demo_action', 10)
        self.cube_pub = self.create_publisher(PoseStamped, '/target_cube_pose', 10)

        self._lock = threading.Lock()
        self._joint = None
        self._image_count = 0
        self._last_image_stamp = 0.0
        self.create_subscription(JointState, '/joint_states', self._on_joint, 10)
        self.create_subscription(Image, '/image_raw', self._on_image, 10)

        self.cube_xy = None

    # ---------- 受信 ----------
    def _on_joint(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(n in pos for n in K.ARM_JOINT_NAMES):
            with self._lock:
                self._joint = np.array([pos[n] for n in K.ARM_JOINT_NAMES])

    def _on_image(self, msg):
        with self._lock:
            self._image_count += 1
            self._last_image_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def joint(self):
        with self._lock:
            return None if self._joint is None else self._joint.copy()

    # ---------- 時間 ----------
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def sim_sleep(self, seconds):
        end = self.now_s() + seconds
        while rclpy.ok() and self.now_s() < end:
            time.sleep(0.005)

    # ---------- 送信 ----------
    def send(self, q_arm, gripper=K.GRIPPER_OPEN):
        stamp = self.get_clock().now().to_msg()

        jt = JointTrajectory()
        jt.joint_names = K.ARM_JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q_arm]
        pt.time_from_start.nanosec = int(0.1 * 1e9)
        jt.points.append(pt)
        self.arm_pub.publish(jt)

        jg = JointTrajectory()
        jg.joint_names = [K.GRIPPER_JOINT_NAME]
        pg = JointTrajectoryPoint()
        pg.positions = [float(gripper)]
        pg.time_from_start.nanosec = int(0.1 * 1e9)
        jg.points.append(pg)
        self.gripper_pub.publish(jg)

        act = JointState()
        act.header.stamp = stamp
        act.name = K.ARM_JOINT_NAMES + [K.GRIPPER_JOINT_NAME]
        act.position = [float(v) for v in q_arm] + [float(gripper)]
        self.action_pub.publish(act)

        if self.cube_xy is not None:
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = 'base_link'
            ps.pose.position.x, ps.pose.position.y = self.cube_xy
            ps.pose.position.z = cube.CUBE_SIZE / 2
            ps.pose.orientation.w = 1.0
            self.cube_pub.publish(ps)

    def move(self, q_from, q_to, duration):
        """min-jerk で q_from → q_to を duration 秒（シミュレーション時間）かけて動かす."""
        t0 = self.now_s()
        period = 1.0 / CONTROL_HZ
        k = 0
        while rclpy.ok():
            t_target = t0 + k * period
            while rclpy.ok() and self.now_s() < t_target:
                time.sleep(0.002)
            s = (self.now_s() - t0) / duration if duration > 0 else 1.0
            self.send(K.min_jerk(q_from, q_to, s))
            if s >= 1.0:
                break
            k += 1

    def hold(self, q, duration):
        self.move(q, q, duration)

    def wait_new_images(self, after_stamp, count=2, timeout=60.0):
        """after_stamp（シミュ時刻）より新しい画像が count 枚届くまで待つ."""
        t_wall = time.time()
        with self._lock:
            start = self._image_count
        while rclpy.ok() and time.time() - t_wall < timeout:
            with self._lock:
                n_new = self._image_count - start
                fresh = self._last_image_stamp > after_stamp
            if n_new >= count and fresh:
                return True
            time.sleep(0.05)
        return False

    # ---------- 1エピソード ----------
    def run_episode(self, idx, out_dir, rng):
        hover = self.get_parameter('hover').value
        approach = self.get_parameter('approach').value
        home_noise = self.get_parameter('home_noise').value

        # 1. 初期姿勢へ（少しばらつかせる）
        q_now = self.joint()
        q_home = HOME + rng.uniform(-home_noise, home_noise, size=4)
        self.move(q_now, q_home, 2.0)
        self.hold(q_home, 0.5)

        # 2. 立方体をランダム配置 → 新しい画像が届くまで待つ
        x, y = cube.sample_position(rng)
        yaw = rng.uniform(-math.pi / 4, math.pi / 4)
        t_set = self.now_s()
        if not cube.set_cube_pose(x, y, yaw):
            self.get_logger().error('立方体の配置に失敗しました。Gazebo が動いているか確認してください。')
            return None
        self.cube_xy = (x, y)
        self.hold(q_home, 1.0)
        if not self.wait_new_images(t_set + 0.5):
            self.get_logger().warn('カメラ画像が更新されません（/image_raw）。このエピソードは飛ばします。')
            return None

        # 逆運動学で目標姿勢を計算
        top = np.array(cube.cube_top_center(x, y))
        p_goal = top + np.array([0.0, 0.0, hover])
        p_pre = p_goal + np.array([0.0, 0.0, approach])
        q_pre, e_pre = K.inverse(p_pre)
        q_goal, e_goal = K.inverse(p_goal, q_init=q_pre)
        if max(e_pre, e_goal) > 0.005:
            self.get_logger().warn(f'逆運動学の誤差が大きいので飛ばします（{e_goal*1000:.1f} mm）')
            return None

        # 3. 記録開始
        bag_path = os.path.join(out_dir, f'ep{idx:03d}')
        proc = subprocess.Popen(['ros2', 'bag', 'record', '-o', bag_path, *RECORD_TOPICS],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3.0)  # レコーダが各トピックを購読し終えるまで待つ（実時間）

        # 4. 手を伸ばす（速さを少しばらつかせる）
        self.hold(q_home, 0.5)
        self.move(q_home, q_pre, rng.uniform(2.0, 3.0))
        self.move(q_pre, q_goal, rng.uniform(1.0, 1.6))
        self.hold(q_goal, self.get_parameter('final_hold').value)

        # 5. 記録停止と結果の保存
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

        q_final = self.joint()
        tip, _ = K.forward(q_final)
        dist = float(np.linalg.norm(tip - p_goal))
        result = {
            'episode': idx, 'bag': bag_path,
            'cube_xy': [x, y], 'cube_yaw': yaw,
            'goal': p_goal.tolist(), 'final_tip': tip.tolist(),
            'final_dist_m': dist,
        }
        self.cube_xy = None
        return result

    def run(self):
        out_dir = self.get_parameter('out_dir').value
        n = self.get_parameter('num_episodes').value
        start = self.get_parameter('start_index').value
        rng = random.Random(self.get_parameter('seed').value + start)
        np_rng = np.random.default_rng(self.get_parameter('seed').value + start)
        os.makedirs(out_dir, exist_ok=True)

        self.get_logger().info('関節状態とシミュレーション時刻を待っています...')
        while rclpy.ok() and (self.joint() is None or self.now_s() == 0.0):
            time.sleep(0.1)

        class _Rng:  # random と numpy の両方のメソッドを使えるように束ねる
            uniform = staticmethod(lambda a, b, size=None:
                                   np_rng.uniform(a, b, size) if size else rng.uniform(a, b))

        log_path = os.path.join(out_dir, 'episodes.jsonl')
        idx = start
        done = 0
        while rclpy.ok() and done < n:
            if os.path.exists(os.path.join(out_dir, f'ep{idx:03d}')):
                idx += 1
                continue
            self.get_logger().info(f'--- エピソード {idx} ({done + 1}/{n}) ---')
            res = self.run_episode(idx, out_dir, _Rng)
            if res is not None:
                with open(log_path, 'a') as f:
                    f.write(json.dumps(res) + '\n')
                self.get_logger().info(f'記録完了: 手先と目標の距離 {res["final_dist_m"]*1000:.1f} mm')
                done += 1
            idx += 1
        self.move(self.joint(), HOME, 2.0)
        self.get_logger().info(f'終了しました。保存先: {out_dir}')


def main():
    rclpy.init()
    node = ScriptedDemo()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
