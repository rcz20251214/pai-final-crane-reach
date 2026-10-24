"""CRANE+ V2 の順運動学・逆運動学（numpyのみで動く軽量版）.

関節の位置・回転軸は rt-net/crane_plus (humble) の
crane_plus_description/urdf/crane_plus.xacro から転記している。
base_link 座標系（ロボット土台の中心、机の上面と同じ高さ）で計算する。
"""

import math

import numpy as np

ARM_JOINT_NAMES = [
    'crane_plus_joint1',
    'crane_plus_joint2',
    'crane_plus_joint3',
    'crane_plus_joint4',
]
GRIPPER_JOINT_NAME = 'crane_plus_joint_hand'

# (親リンクから見た関節の位置[m], 回転軸)
_CHAIN = [
    (np.array([0.0, 0.0, 0.0454]), np.array([0.0, 0.0, 1.0])),   # joint1
    (np.array([0.0, 0.0, 0.026]), np.array([0.0, 1.0, 0.0])),    # joint2
    (np.array([0.0, 0.0, 0.083]), np.array([0.0, -1.0, 0.0])),   # joint3
    (np.array([0.0, 0.0, 0.0935]), np.array([0.0, -1.0, 0.0])),  # joint4
]

_SERVO_HOME = math.radians(150.0)
# URDFの関節制限 [rad]（少し内側に余裕を持たせる）
JOINT_LIMITS = np.array([
    [math.radians(0.0) - _SERVO_HOME, math.radians(300.0) - _SERVO_HOME],
    [math.radians(45.45) - _SERVO_HOME, math.radians(252.20) - _SERVO_HOME],
    [math.radians(3.52) - _SERVO_HOME, math.radians(290.62) - _SERVO_HOME],
    [math.radians(44.57) - _SERVO_HOME, math.radians(251.32) - _SERVO_HOME],
]) + np.array([[0.05, -0.05]])

GRIPPER_OPEN = -0.5236   # -30 deg（physicalai_teleop と同じ値）
GRIPPER_CLOSE = 0.1745   # 10 deg

# 手先点：joint4 のリンク座標系での位置。
# 0.045 がハンドの付け根、指先はそこからさらに数cm先。
DEFAULT_TOOL_OFFSET = np.array([0.0, 0.0, 0.09])


def _rot(axis, angle):
    """Rodrigues の公式で回転行列を作る."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def forward(q, tool_offset=DEFAULT_TOOL_OFFSET):
    """関節角 q[4] から (手先位置[3], 手先の向き=joint4リンクのz軸[3]) を返す."""
    R = np.eye(3)
    p = np.zeros(3)
    for (offset, axis), angle in zip(_CHAIN, q):
        p = p + R @ offset
        R = R @ _rot(axis, angle)
    tip = p + R @ tool_offset
    return tip, R[:, 2]


def _residual(q, target, tool_offset, down_weight):
    tip, z_axis = forward(q, tool_offset)
    pos_err = tip - target
    # 手先が真下（-z）を向くほど 0 になる項。位置より弱く効かせる。
    ori_err = down_weight * (z_axis - np.array([0.0, 0.0, -1.0]))
    return np.concatenate([pos_err, ori_err])


def inverse(target, tool_offset=DEFAULT_TOOL_OFFSET, down_weight=0.03,
            q_init=None, iters=60):
    """手先位置 target[3] を実現する関節角を減衰最小二乗法で求める.

    戻り値: (q[4], 位置誤差[m])。複数の初期値から試して最良の解を返す。
    """
    target = np.asarray(target, dtype=float)
    yaw = math.atan2(target[1], target[0])
    inits = [] if q_init is None else [np.asarray(q_init, dtype=float)]
    for q2, q3, q4 in [(0.6, -0.8, -1.2), (1.0, -1.2, -0.6),
                       (0.3, -0.3, -1.5), (1.2, -0.4, -1.0)]:
        inits.append(np.array([yaw, q2, q3, q4]))

    best_q, best_err = None, float('inf')
    for q in inits:
        q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        # 1段目: 位置＋「下向き」の好み、2段目: 位置だけを合わせ込む
        q = _dls(q, target, tool_offset, down_weight, iters)
        q = _dls(q, target, tool_offset, 0.0, iters)
        err = float(np.linalg.norm(forward(q, tool_offset)[0] - target))
        if err < best_err:
            best_q, best_err = q.copy(), err
    return best_q, best_err


def _dls(q, target, tool_offset, down_weight, iters, lam=1e-3):
    """減衰最小二乗法（数値ヤコビアン）で関節角を更新する."""
    for _ in range(iters):
        r = _residual(q, target, tool_offset, down_weight)
        J = np.zeros((len(r), 4))
        eps = 1e-6
        for j in range(4):
            dq = np.zeros(4)
            dq[j] = eps
            J[:, j] = (_residual(q + dq, target, tool_offset, down_weight) - r) / eps
        step = np.linalg.solve(J.T @ J + lam * np.eye(4), -J.T @ r)
        q = np.clip(q + step, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        if np.linalg.norm(step) < 1e-6:
            break
    return q


def min_jerk(q_start, q_goal, s):
    """0<=s<=1 の進行度で、滑らかに q_start→q_goal を補間する."""
    s = min(max(s, 0.0), 1.0)
    w = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    return (1.0 - w) * np.asarray(q_start) + w * np.asarray(q_goal)
