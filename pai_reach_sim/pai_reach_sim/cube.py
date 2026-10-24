"""赤い立方体（target_cube）の配置を扱う関数と、手動用のコマンド.

座標は 2 系統ある:
  - base 座標: ロボット土台(base_link)基準。学習・評価はこちらで考える。
  - world 座標: Gazebo のワールド基準。base 座標に (0, 0, ROBOT_Z) を足したもの。

使い方（手動で動かす）:
  ros2 run pai_reach_sim randomize_cube             # ランダムな位置へ
  ros2 run pai_reach_sim randomize_cube 0.18 0.05   # base 座標 (x, y) を指定
"""

import math
import os
import random
import subprocess
import sys

CUBE_NAME = 'target_cube'
CUBE_SIZE = 0.04          # [m] ワールドファイルと合わせる
ROBOT_Z = 1.015           # [m] ロボットを置いた高さ（机の上面）
WORLD_NAME = 'default'

# 立方体を置く範囲（base 座標の極座標）。カメラスタンドとロボット土台を避ける。
RADIUS_RANGE = (0.12, 0.21)
ANGLE_RANGE_DEG = (-50.0, 50.0)


def sample_position(rng=random):
    """作業範囲内のランダムな位置 (x, y) [base 座標] を返す."""
    r = rng.uniform(*RADIUS_RANGE)
    a = math.radians(rng.uniform(*ANGLE_RANGE_DEG))
    return r * math.cos(a), r * math.sin(a)


def set_cube_pose(x, y, yaw=0.0, timeout_ms=5000, retries=3):
    """Gazebo のサービスで立方体を base 座標 (x, y) に置く。成功なら True.

    CPU 描画で負荷が高いと応答が遅れることがあるので、何回か試す。
    """
    for _ in range(retries):
        if _set_cube_pose_once(x, y, yaw, timeout_ms):
            return True
    return False


def _set_cube_pose_once(x, y, yaw, timeout_ms):
    z_world = ROBOT_Z + CUBE_SIZE / 2 + 0.002  # 少しだけ浮かせて落とす
    qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
    req = (f'name: "{CUBE_NAME}", '
           f'position: {{x: {x:.4f}, y: {y:.4f}, z: {z_world:.4f}}}, '
           f'orientation: {{x: 0, y: 0, z: {qz:.4f}, w: {qw:.4f}}}')
    cmd = ['ign', 'service', '-s', f'/world/{WORLD_NAME}/set_pose',
           '--reqtype', 'ignition.msgs.Pose',
           '--reptype', 'ignition.msgs.Boolean',
           '--timeout', str(timeout_ms),
           '--req', req]
    try:
        # Gazebo 側（sim.launch.py）と同じく通信先を 127.0.0.1 に揃える
        env = {**os.environ, 'IGN_IP': os.environ.get('IGN_IP', '127.0.0.1')}
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout_ms / 1000 + 5, env=env)
    except subprocess.TimeoutExpired:
        return False
    return 'true' in out.stdout


def cube_top_center(x, y):
    """立方体の上面中心 [base 座標]."""
    return x, y, CUBE_SIZE


def main():
    if len(sys.argv) >= 3:
        x, y = float(sys.argv[1]), float(sys.argv[2])
    else:
        x, y = sample_position()
    yaw = random.uniform(-math.pi / 4, math.pi / 4)
    ok = set_cube_pose(x, y, yaw)
    print(f'cube -> base (x={x:.3f}, y={y:.3f}), yaw={math.degrees(yaw):.0f}deg : '
          f'{"OK" if ok else "FAILED"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
