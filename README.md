# PAI最終課題：カメラで見て手を伸ばす CRANE+ V2

Gazebo 上のロボットアーム CRANE+ V2 が、天井カメラの画像から赤い立方体の位置を判断し、その真上へ手を伸ばすタスクを、模倣学習（ACT）で実現しました。立方体は試行ごとにランダムな位置に置かれます。最終モデルは、学習に使っていない20か所での評価で **成功率75%**（目標から2cm以内、評価時間8秒）でした。

東大松尾・岩澤研 Physical AI講座「知能ロボットシステムエンジニアリング 第5回 ROS 2入門（2/2）」のパイプライン（データ収集 → LeRobotDataset 変換 → ACT 学習 → Gazebo デプロイ → デバッグ）をベースに、次の点を発展させています。

- 観測に**天井カメラの画像**を追加（教材は関節角度のみ）
- 立方体の位置を**試行ごとにランダム化**
- 逆運動学による**お手本動作の自動生成**でデモを収集
- GPU のない Docker 環境でも動く**軽量なシミュレーション起動**
- シミュレータで成功率を測る**閉ループ評価**と、失敗原因の分析にもとづく改善

動画：https://youtu.be/Ypjbgfm7LS0

## 結果

同じ20か所（学習データとは別の乱数で配置）で評価しました。

| # | 条件 | デモ数 | 評価時間 | 2cm以内 | 3cm以内 | 5cm以内 | 最終距離の平均 |
|---|---|---|---|---|---|---|---|
| A | 画像なし（教材と同じ構成） | 60 | 8秒 | 0% | 0% | 35% | 109.7 mm |
| B | 画像あり | 60 | 8秒 | 15% | 40% | 85% | 35.6 mm |
| C | 画像あり＋時間的アンサンブル | 60 | 8秒 | 30% | 70% | 90% | 26.9 mm |
| D | C と同じモデル（原因の確認用） | 60 | 5.5秒 | 60% | 85% | 100% | 20.4 mm |
| E | **最終版**（到達後4秒静止のデモを追加） | 200 | 8秒 | **75%** | 85% | 100% | **17.2 mm** |

- 画像なしのモデルは、立方体の位置に関係なくほぼ同じ一点で手を止めました（A）。
- C では「一度2cm以内に届いたのに、最後には離れていく」試行が多くありました。お手本は到達後1秒で記録が終わるため、それ以降が学習データの外になることが原因でした（D で確認）。
- 到達後も4秒止まり続けるお手本を加えて再学習し、評価時間8秒のまま成功率75%になりました（E）。

詳しい数値は [results/README.md](results/README.md) にあります。

## 構成

```
pai_reach_sim/            ROS 2 パッケージ（シミュレーション・デモ収集・評価）
  launch/sim.launch.py    軽量起動（MoveIt/RViz なし、Gazebo 画面なし、カメラ縮小）
  worlds/reach_table.sdf  机＋赤い立方体のワールド
  pai_reach_sim/
    kinematics.py         CRANE+ V2 の順運動学・逆運動学
    cube.py               立方体の配置（randomize_cube コマンド）
    scripted_demo.py      お手本動作の自動生成と rosbag 記録
    evaluate.py           学習した方策の成功率をシミュレータで測る
training/
  features.py             LeRobotDataset の特徴量定義（画像あり / なし）
  convert.py              rosbag → LeRobotDataset
  train.py                ACT の学習（GPU）
  debug_policy.py         オフライン評価（予測と正解の比較、全エピソード集計）
  infer_node.py           Gazebo 上で方策を動かす推論ノード
notebooks/
  train_on_colab.ipynb    Colab（L4）で学習・オフライン評価
results/
  README.md               評価結果の詳細
```

## 必要な環境

- 教材の Docker イメージ `airobotbook/ros2-desktop-ai-robot-book-humble`
- [rt-net/crane_plus](https://github.com/rt-net/crane_plus) の **humble ブランチ**（カメラ対応版）を `~/pai_ws/src` に入れてビルド済みであること
  （イメージに入っている `~/airobot_ws` の CRANE+ は古く、カメラがありません）
- 教材の学習環境 [matsuolab/pai_training](https://github.com/matsuolab/pai_training)（pixi）を `~/pai_training` にセットアップ済みであること
- 学習用の GPU（Google Colab の L4 で確認）

## 導入

```bash
cd ~/pai_ws/src
git clone -b humble https://github.com/rt-net/crane_plus.git
git clone <このリポジトリの URL>
cd ~/pai_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## 使い方

### 1. シミュレーションを起動

```bash
ros2 launch pai_reach_sim sim.launch.py viewer:=true
```

| 引数 | 既定値 | 内容 |
|---|---|---|
| `gui` | `false` | Gazebo の画面を表示する（重い。録画時のみ推奨） |
| `viewer` | `false` | カメラ映像を rqt_image_view で表示する |
| `image_width` / `image_height` | `224` / `168` | カメラ画像のサイズ |
| `camera_rate` | `10` | カメラの撮影周期 [Hz] |
| `render_engine` | `ogre2` | `ogre` にすると CPU 描画で速くなる場合がある |

立方体を手で動かす場合：

```bash
ros2 run pai_reach_sim randomize_cube            # ランダムな位置へ
ros2 run pai_reach_sim randomize_cube 0.18 0.05  # ロボット基準の (x, y) [m] を指定
```

### 2. お手本デモを自動収集

最終版（E）は、到達後1秒静止の60本と、4秒静止の140本を合わせた200本で学習しました。

```bash
ros2 run pai_reach_sim scripted_demo --ros-args \
    -p num_episodes:=60 -p out_dir:=/home/ubuntu/dataset/rosbags/reach
ros2 run pai_reach_sim scripted_demo --ros-args \
    -p num_episodes:=140 -p start_index:=60 -p final_hold:=4.0 \
    -p out_dir:=/home/ubuntu/dataset/rosbags/reach
```

- 1エピソードごとに `ep000/`, `ep001/`, … の rosbag と、結果の記録 `episodes.jsonl` が保存されます。
- 途中で止めても、同じコマンドで続きの番号から再開します。
- `final_hold`：目標に着いた後、止まったまま記録し続ける時間 [秒]。

### 3. LeRobotDataset へ変換（pai_training の pixi 環境）

```bash
cd ~/pai_training
R=~/pai_ws/src/pai-final-crane-reach/training
pixi run python $R/convert.py --bags ~/dataset/rosbags/reach \
    --out ~/dataset/lerobot_dataset/crane_reach_image_v2 --repo-id crane_reach_image_v2 --mode image
```

- `--mode image`：関節角度＋カメラ画像（提案手法）
- `--mode state`：関節角度のみ（教材と同じ構成。比較用）
- 画像と関節角度は、メッセージのヘッダ時刻（シミュレーション時間）で 10Hz に揃えます。CPU 描画では画像が実時間で遅れて届くためです。
- 関節角度は、メッセージ内の並び順に頼らず関節名で対応づけ、観測と行動で列の順番を統一しています。

### 4. Colab で学習

1. 変換したデータセットを Google Drive の `pai_final/datasets/` に置き、このリポジトリを `pai_final/` に置きます。
2. `notebooks/train_on_colab.ipynb` を Colab で開き、L4 GPU で上から実行します。

最終版（E）の学習設定：

```bash
python training/train.py \
    --dataset-root /content/data/crane_reach_image_v2 --repo-id crane_reach_image_v2 \
    --out <保存先>/act_image_v2 --steps 30000 --batch-size 32 --val-episodes 10
```

LeRobot は教材と同じコミット（768e366）を使い、`datasets==3.6.0`、`draccus==0.10.0` に固定しています。新しい版だと、データの読み込みやモデル設定の保存・読み込みが失敗します。

### 5. 推論と評価

学習済みモデル（`last` フォルダ）を `~/checkpoints/act_image_v2/` に置きます。

```bash
# ターミナル1：シミュレーション
ros2 launch pai_reach_sim sim.launch.py viewer:=true

# ターミナル2：推論ノード（ROS を source してから pixi 環境で）
source ~/pai_ws/install/setup.bash
cd ~/pai_training
pixi run python ~/pai_ws/src/pai-final-crane-reach/training/infer_node.py \
    --model-path ~/checkpoints/act_image_v2/last --no-autostart --temporal-ensemble 0.01

# ターミナル3：成功率の測定
source ~/pai_ws/install/setup.bash
ros2 run pai_reach_sim evaluate --ros-args \
    -p num_trials:=20 -p eval_dir:=/home/ubuntu/eval/act_image_v2_te -p label:=act_image_v2_te
```

- 成功は「方策を `trial_time`（既定8秒）動かした後、手先が目標（立方体の上面から2cm上）から2cm以内」。
- 結果は `trials.csv`（試行ごと）と `summary.json`（集計）に保存されます。2cm・3cm・5cmでの成功率、途中で一度でも届いた割合、水平・高さ方向の誤差も記録します。
- `--temporal-ensemble`：毎ステップ推論し、重なった予測を指数重みで平均します（ACT 論文の手法）。省略すると、学習時の `n_action_steps` ごとにまとめて実行します。
- 立方体の位置は `eval_seed`（既定1000）で決まるので、同じ値なら全モデルを同じ配置で比べられます。

## ライセンスとクレジット

このリポジトリのコードは Apache License 2.0 です（[LICENSE](LICENSE)）。

- ロボットモデル・Gazebo 環境：[rt-net/crane_plus](https://github.com/rt-net/crane_plus)（Apache-2.0）。起動ファイルとワールドファイルは、その `crane_plus_gazebo` をもとにしています。
- 学習パイプライン：[matsuolab/pai_training](https://github.com/matsuolab/pai_training)、[matsuolab/physicalai_teleop](https://github.com/matsuolab/physicalai_teleop)。`training/` の各スクリプトは、教材のスクリプトをもとに拡張しています。
- 模倣学習フレームワーク：[huggingface/lerobot](https://github.com/huggingface/lerobot)（Apache-2.0）
- ACT：T. Z. Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023
