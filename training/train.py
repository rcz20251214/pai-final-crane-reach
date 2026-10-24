#!/usr/bin/env python3
"""ACT（Action Chunking with Transformer）を学習する. GPU（Colab L4 など）を想定.

教材の pai_training/scripts/train.py（HuggingFace LeRobot の examples/3_train_policy.py が元）をもとに、
  - 天井カメラ画像を入力に使えるようにした（データセットに画像があれば自動で使う）
  - 最後の数エピソードを検証用に取り分け、検証損失も記録する
  - GPU を自動で使い、一定間隔でチェックポイントを保存する
  - wandb は任意（--wandb を付けたときだけ使う）

使い方:
  python train.py --dataset-root /content/drive/MyDrive/pai/crane_reach_image \
      --repo-id crane_reach_image --out /content/drive/MyDrive/pai/ckpt/act_image --steps 20000
"""

import argparse
import json
import time
from pathlib import Path

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--out", required=True, help="学習済みモデルの保存先")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5, help="LeRobot の ACT 既定値と同じ")
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--chunk-size", type=int, default=20, help="1回の推論で出す行動の数（fps=10なら2秒分）")
    ap.add_argument("--n-action-steps", type=int, default=5, help="推論1回あたり実行する行動の数")
    ap.add_argument("--val-episodes", type=int, default=5, help="検証用に取り分けるエピソード数（末尾から）")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-freq", type=int, default=100)
    ap.add_argument("--val-freq", type=int, default=1000)
    ap.add_argument("--save-freq", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true", help="wandb に記録する")
    ap.add_argument("--no-pretrained-backbone", action="store_true",
                    help="ResNet の ImageNet 事前学習済み重みを使わない（動作テスト用）")
    return ap.parse_args()


@torch.no_grad()
def evaluate(policy, loader, device, max_batches=50):
    policy.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        loss, _ = policy.forward(batch)
        total += loss.item()
        n += 1
    policy.train()
    return total / max(n, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    features = dataset_to_policy_features(meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}
    print(f"入力: { {k: tuple(v.shape) for k, v in input_features.items()} }")
    print(f"出力: { {k: tuple(v.shape) for k, v in output_features.items()} }")

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        optimizer_lr=args.lr,
        optimizer_lr_backbone=args.lr_backbone,
    )
    if args.no_pretrained_backbone:
        cfg.pretrained_backbone_weights = None
    policy = ACTPolicy(cfg, dataset_stats=meta.stats).to(device)
    policy.train()

    # 行動は「今から chunk_size 個先まで」を正解にする
    delta_timestamps = {"action": [i / meta.fps for i in range(args.chunk_size)]}
    all_eps = list(range(meta.total_episodes))
    n_val = min(args.val_episodes, max(len(all_eps) - 1, 0))
    train_eps, val_eps = all_eps[: len(all_eps) - n_val], all_eps[len(all_eps) - n_val:]
    print(f"学習 {len(train_eps)} エピソード / 検証 {len(val_eps)} エピソード")

    # 注意: この版の LeRobot は episodes= で一部だけ読むと、0番以外から始まる場合に
    # 番号がずれて落ちる。全体を読んでからフレーム番号で学習用／検証用に分ける。
    full_ds = LeRobotDataset(args.repo_id, root=args.dataset_root, delta_timestamps=delta_timestamps)

    def frames_of(eps):
        idx = []
        for e in eps:
            idx += range(full_ds.episode_data_index["from"][e].item(),
                         full_ds.episode_data_index["to"][e].item())
        return idx

    train_ds = torch.utils.data.Subset(full_ds, frames_of(train_eps))
    print(f"学習フレーム数: {len(train_ds)}")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0)
    val_loader = None
    if val_eps:
        val_ds = torch.utils.data.Subset(full_ds, frames_of(val_eps))
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ACT の慣例どおり、画像バックボーン（ResNet）だけ学習率を下げる
    params = [
        {"params": [p for n, p in policy.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad]},
        {"params": [p for n, p in policy.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad],
         "lr": args.lr_backbone},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    if args.wandb:
        import wandb
        wandb.init(project="pai_crane_reach", config=vars(args))

    history = []
    step, t0 = 0, time.time()
    while step < args.steps:
        for batch in train_loader:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % args.log_freq == 0:
                rate = step / (time.time() - t0)
                print(f"step {step:6d}  loss {loss.item():.4f}  ({rate:.1f} step/s)")
                rec = {"step": step, "train_loss": loss.item(), **loss_dict}
                if val_loader is not None and step % args.val_freq == 0:
                    rec["val_loss"] = evaluate(policy, val_loader, device)
                    print(f"           val_loss {rec['val_loss']:.4f}")
                history.append(rec)
                if args.wandb:
                    wandb.log(rec, step=step)
            if step % args.save_freq == 0 or step >= args.steps:
                policy.save_pretrained(out / f"step_{step:06d}")
                policy.save_pretrained(out / "last")
            if step >= args.steps:
                break

    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "train_args.json").write_text(json.dumps(
        {**vars(args), "train_episodes": train_eps, "val_episodes": val_eps}, indent=1))
    print(f"保存しました: {out / 'last'}")
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
