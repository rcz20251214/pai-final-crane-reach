#!/usr/bin/env python3
"""scripted_demo で記録した rosbag を LeRobotDataset に変換する.

教材の scripts/rosbag2lerobot.py との違い:
  - 天井カメラ画像（/image_raw）を観測に追加できる（--mode image）
  - 受信時刻ではなくメッセージのヘッダ時刻（シミュレーション時間）で同期する。
    CPU描画では画像が実時間で遅れて届くため、受信時刻で揃えると画像と関節角度がずれる。
  - 関節は並び順ではなく名前で取り出す
  - エピソードを最長に合わせて水増しせず、実際の長さで保存する

使い方（pai_training の pixi 環境で実行）:
  cd ~/pai_training
  pixi run python ~/pai_ws/src/pai-final-crane-reach/training/convert.py \
      --bags ~/dataset/rosbags/reach \
      --out ~/dataset/lerobot_dataset/crane_reach_image --repo-id crane_reach_image --mode image
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import IMAGE_KEY, ROBOT_TYPE, TASK, make_features  # noqa: E402

ARM = ["crane_plus_joint1", "crane_plus_joint2", "crane_plus_joint3", "crane_plus_joint4"]
HAND = "crane_plus_joint_hand"
ORDER = ARM + [HAND]


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def read_episode(bag_dir: Path, want_image: bool):
    """1エピソード分のメッセージを (時刻[ns], 値) の列として読み出す."""
    topics = {"/joint_states", "/demo_action"} | ({"/image_raw"} if want_image else set())
    data = {t: [] for t in topics}
    with Reader(bag_dir) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, _bag_t, raw in reader.messages(connections=conns):
            msg = deserialize_cdr(raw, conn.msgtype)
            t = stamp_ns(msg.header)
            if conn.topic in ("/joint_states", "/demo_action"):
                pos = dict(zip(msg.name, msg.position))
                if not all(n in pos for n in ORDER):
                    continue
                data[conn.topic].append((t, np.array([pos[n] for n in ORDER], dtype=np.float32)))
            else:
                if msg.encoding not in ("rgb8", "bgr8"):
                    raise ValueError(f"未対応の画像形式: {msg.encoding}")
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
                img = img[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
                if msg.encoding == "bgr8":
                    img = img[:, :, ::-1]
                data[conn.topic].append((t, img.copy()))
    for t, seq in data.items():
        if not seq:
            raise ValueError(f"{bag_dir}: トピック {t} が記録されていません")
        seq.sort(key=lambda x: x[0])
    return data


def sample_and_hold(seq, target_ns):
    """各目標時刻について「その時刻以前で最新」の値を取る."""
    times = np.array([t for t, _ in seq])
    idx = np.searchsorted(times, target_ns, side="right") - 1
    idx = np.clip(idx, 0, len(seq) - 1)
    return [seq[i][1] for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bags", required=True, help="ep000/, ep001/, ... を含むフォルダ")
    ap.add_argument("--out", required=True, help="出力する LeRobotDataset のフォルダ")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--mode", choices=["image", "state"], default="image")
    ap.add_argument("--fps", type=int, default=10, help="サンプリング周期 [Hz]（カメラと同じ10を推奨）")
    ap.add_argument("--overwrite", action="store_true", help="出力先が既にあれば消して作り直す")
    args = ap.parse_args()

    # lerobot の import は重いので引数チェック後に行う
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    bag_root = Path(args.bags).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    episodes = sorted(p.parent for p in bag_root.rglob("*.db3"))
    if not episodes:
        sys.exit(f"rosbag が見つかりません: {bag_root}")
    print(f"{len(episodes)} エピソードを変換します（mode={args.mode}, fps={args.fps}）")

    if out.exists():
        if not args.overwrite:
            sys.exit(f"出力先が既にあります: {out}（作り直すなら --overwrite）")
        shutil.rmtree(out)

    want_image = args.mode == "image"
    first = read_episode(episodes[0], want_image)
    hw = first["/image_raw"][0][1].shape[:2] if want_image else (168, 224)
    features = make_features(args.mode, hw)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=out,
        robot_type=ROBOT_TYPE,
        features=features,
        use_videos=False,
        image_writer_threads=4 if want_image else 0,
    )

    period_ns = int(1e9 / args.fps)
    n_frames_total = 0
    for i, ep in enumerate(episodes):
        data = first if i == 0 else read_episode(ep, want_image)
        start = max(seq[0][0] for seq in data.values())
        end = min(seq[-1][0] for seq in data.values())
        targets = np.arange(start, end + 1, period_ns)
        if len(targets) < 5:
            print(f"  {ep.name}: 短すぎるので飛ばします（{len(targets)} フレーム）")
            continue

        states = sample_and_hold(data["/joint_states"], targets)
        actions = sample_and_hold(data["/demo_action"], targets)
        images = sample_and_hold(data["/image_raw"], targets) if want_image else None

        for k in range(len(targets)):
            frame = {
                "observation.state": states[k],
                "action": actions[k],
                "task": TASK,
            }
            if want_image:
                frame[IMAGE_KEY] = images[k]
            else:
                frame["observation.environment_state"] = states[k]
            dataset.add_frame(frame)
        dataset.save_episode()
        n_frames_total += len(targets)
        print(f"  {ep.name}: {len(targets)} フレーム（{(end - start) / 1e9:.1f} 秒）")

    if dataset.image_writer is not None:
        dataset.stop_image_writer()
    print(f"完了: {dataset.num_episodes} エピソード / {n_frames_total} フレーム -> {out}")


if __name__ == "__main__":
    main()
