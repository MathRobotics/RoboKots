# Jv / Jᵀv 高速化の比較（2026-09-23）

## 変更内容

- 通常トルク（motion order 3、時間微分なし）の Rust Jv / Jᵀv に、body 座標系 RNEA の直接微分・逆伝播を追加。密ヤコビ行列や単位行列の種は生成しない。
- トルクの両方向積で同じ局所微分ブロックを共有し、motion と重力が同じ間は再利用する。RNEA 作業領域はリンク数・関節数に比例し、バッチ要素ごとに保持する。
- 運動学では既存の祖先経路アルゴリズムを維持し、座標変換の係数と各経路のスクリュー列を再利用する。motion・出力指定・バッチサイズの変更時に更新する。こちらの保存量は選択リンクの祖先経路数と次数に依存する。
- 高次トルクや力・運動量を含む混在出力の計算経路は今回の高速化対象外。NumPy の計算経路も変更していない。

## 測定条件

合成7自由度の回転関節直列モデル＋固定工具。実機固有モデルではない。float64、seed=20260923、重力=[0.2,-0.3,-9.81]。macOS 15.7.4 arm64、Python 3.13.1、NumPy 2.4.6、Rust release ビルド。Python 公開 API の呼び出し・入出力変換を含み、モデル生成は除外。warmup 5回、15標本×各3回の中央値。ビルド・テストと同時に測定していない。

修正前は今回の作業開始時に同じスクリプトで測定した。修正前後は別プロセス・別時点なので数％の差から改善・退行を断定しない。NumPy/Rust 間のフォールバックを禁止して測定。単一ベクトルの積であり、多列 RHS の性能は未測定。

## Rust の修正前後

単位は µs。state_ready は同じ状態で微分用キャッシュを warmup 済み。including_state は2つの motion を交互に使い、入力更新・状態計算・微分用係数の再計算を含む。

| 出力 | 条件 | 操作 | 修正前 µs | 修正後 µs | 前/後 |
|---|---|---|---:|---:|---:|
| トルク | state_ready | dense | 21.11 | 20.83 | 1.01× |
| トルク | state_ready | jvp | 15.00 | 13.42 | 1.12× |
| トルク | state_ready | vjp | 56.69 | 14.99 | 3.78× |
| トルク | including_state | dense | 30.97 | 31.62 | 0.98× |
| トルク | including_state | jvp | 25.14 | 25.38 | 0.99× |
| トルク | including_state | vjp | 67.49 | 27.17 | 2.48× |
| 位置・速度・加速度（world） | state_ready | dense | 11.36 | 8.56 | 1.33× |
| 位置・速度・加速度（world） | state_ready | jvp | 9.07 | 6.49 | 1.40× |
| 位置・速度・加速度（world） | state_ready | vjp | 9.92 | 7.12 | 1.39× |
| 位置・速度・加速度（world） | including_state | dense | 16.65 | 17.54 | 0.95× |
| 位置・速度・加速度（world） | including_state | jvp | 14.36 | 14.67 | 0.98× |
| 位置・速度・加速度（world） | including_state | vjp | 15.76 | 15.38 | 1.03× |

状態計算済みのトルク Jᵀv は約3.8倍、状態更新込みでも約2.5倍に改善。Jv の改善は小さく、状態更新込みの Jv は両出力ともほぼ同等。運動学の再利用は同じ状態に複数の方向・重みを適用するときに有効で、毎回状態が変わる場合の高速化は確認できていない。

## 正しさと検証

今回の測定で NumPy 解析との最大絶対差は 7.105e-14、最大相対 Frobenius 誤差は 2.200e-15。

- 新規トルクの積のテスト：分岐・固定関節、指定順・重複・部分出力、零/非零重力、ゼロ姿勢、単一/多列 RHS、多次元バッチを既存密解析と比較。中心差分 h=1e-6（独立した状態計算）で Jv/Jᵀv を照合。
- motion の q/v/a と重力変更による RNEA キャッシュ無効化、運動学/トルク切替、空バッチを検証。運動学の motion・出力順・バッチ形状変更も NumPy と照合。
- Python 関連回帰：505 passed、2 deselected。追加の運動学キャッシュ回帰を含む対象ファイルは別実行で10 passed（そのうち9件は前述と重複）。
- Rust 単独実行（Python 機能なし）：7単体＋9統合＋1ドキュメントテストが成功。

```bash
.venv/bin/python -m developer.benchmarks.numpy_rust_jacobian_compare --case torque --serial-dof 7 --output developer/benchmarks/results/rust_products_after.json
.venv/bin/python -m developer.benchmarks.numpy_rust_jacobian_compare --case kinematics --serial-dof 7 --output developer/benchmarks/results/rust_kinematic_products_after.json
cargo test --offline --no-default-features --manifest-path robokots/_rust/Cargo.toml
```

生データ（全測定標本、初回呼出し、環境、拡張バイナリ SHA256、モデル定義）：

- [トルク・修正前](rust_products_before.json) / [修正後](rust_products_after.json)
- [運動学・修正前](rust_kinematic_products_before.json) / [修正後](rust_kinematic_products_after.json)
