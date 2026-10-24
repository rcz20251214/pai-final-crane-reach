"""学習した方策をシミュレータで繰り返し試し、成功率を測る評価ノード.

1試行の流れ:
  1. 方策を止める（/policy/stop）→ アームを初期姿勢へ戻す
  2. 立方体を「学習データとは別の乱数系列」でランダムに配置し、画像の更新を待つ
  3. 方策を動かす（/policy/start）→ 一定時間（シミュレーション時間）待つ
  4. 手先と目標（立方体の上面＋hover）の距離を、関節角度から順運動学で計算して記録

成功の判定: 試行終了時の距離が success_threshold 以下。
結果は eval_dir に trials.csv と summary.json として保存する。

使い方（別ターミナルで infer_node.py を --no-autostart 付きで起動しておく）:
  ros2 run pai_reach_sim evaluate --ros-args -p num_trials:=20 -p eval_dir:=/home/ubuntu/eval/act_image
"""

import csv
import json
import math
import os
import random
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger

from pai_reach_sim import cube
from pai_reach_sim import kinematics as K
from pai_reach_sim.scripted_demo import HOME, ScriptedDemo


class Evaluator(ScriptedDemo):

    def __init__(self):
        super().__init__('policy_evaluator')
        self.declare_parameter('num_trials', 20)
        self.declare_parameter('trial_time', 8.0)          # 方策を動かす時間 [s, シミュ時間]
        self.declare_parameter('success_threshold', 0.02)  # [m]
        self.declare_parameter('eval_seed', 1000)          # 学習データ(seed=0〜)と重ならない値
        self.declare_parameter('label', 'policy')
        self.declare_parameter('eval_dir', os.path.expanduser('~/eval/policy'))
        self.start_cli = self.create_client(Trigger, '/policy/start')
        self.stop_cli = self.create_client(Trigger, '/policy/stop')

    def call(self, client, timeout=10.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'{client.srv_name} が見つかりません。infer_node.py を起動してください。')
        fut = client.call_async(Trigger.Request())
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            time.sleep(0.01)
        return fut.done() and fut.result().success

    def tip_distance(self, goal):
        tip, _ = K.forward(self.joint())
        return float(np.linalg.norm(tip - goal)), tip

    def run_trial(self, idx, rng):
        hover = self.get_parameter('hover').value
        trial_time = self.get_parameter('trial_time').value

        self.call(self.stop_cli)
        self.move(self.joint(), HOME, 2.0)
        self.hold(HOME, 0.5)

        x, y = cube.sample_position(rng)
        yaw = rng.uniform(-math.pi / 4, math.pi / 4)
        t_set = self.now_s()
        if not cube.set_cube_pose(x, y, yaw):
            raise RuntimeError('立方体の配置に失敗しました（Gazebo が動いているか確認してください）')
        self.cube_xy = (x, y)
        self.hold(HOME, 1.0)
        self.wait_new_images(t_set + 0.5)

        goal = np.array(cube.cube_top_center(x, y)) + np.array([0.0, 0.0, hover])
        self.call(self.start_cli)
        t0 = self.now_s()
        min_dist, t_min = float('inf'), 0.0
        while rclpy.ok() and self.now_s() - t0 < trial_time:
            d, _ = self.tip_distance(goal)
            if d < min_dist:
                min_dist, t_min = d, self.now_s() - t0
            time.sleep(0.05)
        final_dist, tip = self.tip_distance(goal)
        self.call(self.stop_cli)
        self.cube_xy = None
        return {
            'trial': idx, 'cube_x': x, 'cube_y': y,
            'final_dist_m': final_dist, 'min_dist_m': min_dist,
            't_min_s': t_min,  # 最も近づいた時刻（方策を動かし始めてから）
            # 水平方向（どこへ手を伸ばしたか）と高さ方向のずれを分けて記録する
            'xy_err_m': float(np.linalg.norm(tip[:2] - goal[:2])),
            'z_err_m': float(tip[2] - goal[2]),
            'tip_x': tip[0], 'tip_y': tip[1], 'tip_z': tip[2],
        }

    def run(self):
        n = self.get_parameter('num_trials').value
        thr = self.get_parameter('success_threshold').value
        out_dir = self.get_parameter('eval_dir').value
        label = self.get_parameter('label').value
        rng = random.Random(self.get_parameter('eval_seed').value)
        os.makedirs(out_dir, exist_ok=True)

        while rclpy.ok() and (self.joint() is None or self.now_s() == 0.0):
            time.sleep(0.1)

        rows = []
        for i in range(n):
            r = self.run_trial(i, rng)
            r['success'] = int(r['final_dist_m'] <= thr)
            rows.append(r)
            self.get_logger().info(
                f'試行 {i + 1}/{n}: 最終距離 {r["final_dist_m"] * 1000:.1f} mm '
                f'(水平 {r["xy_err_m"] * 1000:.1f} / 高さ {r["z_err_m"] * 1000:+.1f}) '
                f'{"成功" if r["success"] else "失敗"}')

        with open(os.path.join(out_dir, 'trials.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        dists = np.array([r['final_dist_m'] for r in rows])
        summary = {
            'label': label, 'trials': n, 'success_threshold_m': thr,
            'success_rate': float(np.mean([r['success'] for r in rows])),
            # 判定の厳しさを変えたときの成功率（レポート用）
            'success_rate_by_threshold': {
                f'{t * 100:.0f}cm': float(np.mean(dists <= t)) for t in (0.02, 0.03, 0.05)},
            'final_dist_mean_m': float(dists.mean()),
            'final_dist_median_m': float(np.median(dists)),
            'min_dist_mean_m': float(np.mean([r['min_dist_m'] for r in rows])),
            # 途中で一度でも閾値内に入った割合（到達後に離れていく現象を見るため）
            'reached_rate_by_threshold': {
                f'{t * 100:.0f}cm': float(np.mean([r['min_dist_m'] <= t for r in rows]))
                for t in (0.02, 0.03, 0.05)},
            'xy_err_mean_m': float(np.mean([r['xy_err_m'] for r in rows])),
            'z_err_mean_m': float(np.mean([r['z_err_m'] for r in rows])),
        }
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=1)
        self.get_logger().info(
            f'[{label}] 成功率 {summary["success_rate"] * 100:.0f}% '
            f'（距離 {thr * 100:.0f}cm 以内）, 最終距離の平均 {summary["final_dist_mean_m"] * 1000:.1f} mm '
            f'（水平 {summary["xy_err_mean_m"] * 1000:.1f} mm, 高さ {summary["z_err_mean_m"] * 1000:+.1f} mm）')
        self.move(self.joint(), HOME, 2.0)


def main():
    rclpy.init()
    node = Evaluator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
