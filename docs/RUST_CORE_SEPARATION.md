# Rustコアの分離

Rust実装・native tests・高速化試作は兄弟リポジトリ `../robokots-rs` に移しました。
RustディレクトリのGit履歴は新しいリポジトリに引き継いでいます。
このリポジトリはPython公開API、NumPy/JAX実装、Python回帰テストと薄いPyO3入口を持ちます。
import名 `robokots._rust_core` とpipプロジェクト名 `robokots` は変わりません。

```text
workspace/
  RoboKots/
  robokots-rs/
```

Python環境を有効にしてRoboKotsルートから `maturin develop --release` でビルドします。
Rust単体は `cargo test --manifest-path ../robokots-rs/Cargo.toml --no-default-features`。
新しい場所にもRoboKotsの既存LICENSEを保持しています。

現在は非公開ローカル開発用のpath依存です。リモート・自動配布は設定していません。
公開sdistにはpath依存のソースが同梱される可能性があるため、非公開期間は公開sdistを作成・配布しないでください。
Cargoはコアと拡張入口の両方を `publish = false` にしています。
公開済みバージョンの内容とGit履歴はこの移行で消去しません。

互換バインディング本体 `py_api.rs` は内部ワークスペースへの依存が強いため、
まずRustコアのoptional `python` feature内に保持しています。
Rustのdefault featureは空で、単体利用はPython不要です。

分離時の確認: Rust単体20テスト、分離先のworld集約32ケース、
新しくビルドしたPython拡張を使う関連回帰テスト227件が通過しました。
