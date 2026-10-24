#!/usr/bin/env python3
"""学習した方策を、データセットの観測に対して動かし、正解の行動と比べる（オフライン評価）.

教材の pai_training/scripts/debug_policy.py を拡張し、
  - 画像入力の方策にも対応
  - 1エピソードだけでなく全エピソードを評価し、学習用／検証用に分けて集計
  - グラフは画面に出さず PNG に保存（Colab・サーバーでも動く）

使い方:
  python debug_policy.py --dataset-root ... --repo-id ... --policy-path .../last --out-dir results/act_image
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.common.policies.act.modeling_act import ACTPolicy  # noqa: E402

JOINTS = ["joint1", "joint2", "joint3", "joint4", "gripper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--plot-episodes", type=int, default=4, help="グラフを保存するエピソード数（検証用から優先）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    policy = ACTPolicy.from_pretrained(args.policy_path).to(args.device)
    policy.eval()
    input_keys = list(policy.config.input_features.keys())

    # train.py が保存した学習／検証の分け方を読む（なければ全部「不明」扱い）
    split = {}
    args_file = Path(args.policy_path).parent / "train_args.json"
    if args_file.exists():
        ta = json.loads(args_file.read_text())
        split = {**{e: "train" for e in ta["train_episodes"]}, **{e: "val" for e in ta["val_episodes"]}}

    val_first = sorted(range(dataset.num_episodes), key=lambda e: split.get(e) != "val")
    to_plot = set(val_first[: args.plot_episodes])

    results = []
    for ep in range(dataset.num_episodes):
        lo = dataset.episode_data_index["from"][ep].item()
        hi = dataset.episode_data_index["to"][ep].item()
        policy.reset()
        preds, gts = [], []
        for i in range(lo, hi):
            s = dataset[i]
            obs = {k: s[k].unsqueeze(0).to(args.device) for k in input_keys}
            with torch.no_grad():
                pred = policy.select_action(obs).squeeze(0).cpu().numpy()
            preds.append(pred)
            gts.append(s["action"].numpy())
        preds, gts = np.array(preds), np.array(gts)
        err = np.linalg.norm(preds - gts, axis=1)
        rec = {
            "episode": ep,
            "split": split.get(ep, "unknown"),
            "mean_l2": float(err.mean()),
            "final_l2": float(err[-1]),
            "per_joint_mae": np.abs(preds - gts).mean(axis=0).round(5).tolist(),
        }
        results.append(rec)
        print(f"ep{ep:03d} [{rec['split']:5s}] 平均L2 {rec['mean_l2']:.4f}  最終L2 {rec['final_l2']:.4f}")

        if ep in to_plot:
            fig, axes = plt.subplots(len(JOINTS), 1, figsize=(8, 10), sharex=True)
            for j, ax in enumerate(axes):
                ax.plot(gts[:, j], "--", label="GT")
                ax.plot(preds[:, j], label="pred")
                ax.set_ylabel(JOINTS[j])
            axes[0].legend(loc="upper right")
            axes[-1].set_xlabel("frame")
            fig.suptitle(f"episode {ep} ({rec['split']})  mean L2={rec['mean_l2']:.4f}")
            fig.tight_layout()
            fig.savefig(out / f"episode_{ep:03d}.png", dpi=110)
            plt.close(fig)

    summary = {}
    for name in ("train", "val", "unknown"):
        sub = [r for r in results if r["split"] == name]
        if sub:
            summary[name] = {
                "episodes": len(sub),
                "mean_l2": float(np.mean([r["mean_l2"] for r in sub])),
                "final_l2": float(np.mean([r["final_l2"] for r in sub])),
            }
    (out / "offline_eval.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
    print("\n=== 集計 ===")
    for name, s in summary.items():
        print(f"{name:7s}: {s['episodes']} エピソード  平均L2 {s['mean_l2']:.4f}  最終L2 {s['final_l2']:.4f}")
    print(f"保存先: {out}")


if __name__ == "__main__":
    main()
