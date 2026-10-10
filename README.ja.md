# LLM Long-Term Memory

LLM に長期記憶を与えるレイヤー。**ENGRAM v2.1**（[`SPEC.md`](SPEC.md)）を実装しています。人間の記憶の原理から導いた**痕跡（trace）モデル**で、**ローカルの埋め込みモデル**（EmbeddingGemma の GGUF を llama.cpp で実行。CUDA があれば GPU）と**単一ファイルの SQLite** だけで動きます。Rust 製で、エージェントハーネスに組み込める依存の少ないライブラリ（`engram`）と、アプリ `ltm`（Web UI・ヘッドレス実行・ベンチマーク）から成ります。LLM が生成するのは 書込み・利用（引用）・オフラインの**夢**（統合）の 3 点だけです。

同じアルゴリズムを Dart パッケージ（[`../long-term-memory`](../long-term-memory)）としても提供しています。演算（float32 ベクトル、コサインは次元順に f64 で加算）を共有しているため、言語間一致シナリオと 220 ステップのシード付きワークロードの両方で Dart のトレースを**ビット単位で**再現します。

---

## モデルの全体像

記憶は**痕跡**です。持つのは 2 つの数 — 最後に想起した時刻と安定度（半減期）— と 1 つのフラグ（`consolidated`）だけ。

```text
R(now)   = 2^(−max(0, now − last_recall) / stability)          想起可能性 ∈ [0,1]
strength = stability · R                                       将来の想起可能性の総量
a        = max(0, (cos − cosine_floor) / (1 − cosine_floor))   手がかりの活性化
想起      : stability ← min(stability · (1 + gain·a·(1−R)), S_max);  last_recall ← now
新規      : stability = clamp(S0 · salience, 1 s, S_max)
```

| 動詞 | 動作 |
|---|---|
| `remember(text, salience, cue)` | 同一テキストはリハーサル。それ以外は挿入し、**決して上書きしない**。`cue` が近傍（cos ≥ θ_related）に届く痕跡は*不安定*に生まれる。容量超過時は猶予期間（3 日）外で strength 最小の痕跡を忘却。 |
| `recall(query)` | 複数手がかりのコサイン → `score = a·(α + (1−α)·R)` → 絶対・相対閾値 → MMR → `[unix tz] text 《id》` を ≤1024 字で注入。注入は「露出」なので半分だけ強化。`peek(query)` は副作用なしの同じ検索。 |
| `cite(reply)` | LLM が引用した《id》の記憶を「使用」として完全に強化。 |
| `forget(id)` | id 指定の物理削除。 |
| `dream(adjudicator, budget)` | 不安定な痕跡（先入れ先出し、≤ 8·budget）を種に、その `cue` が再活性化した**より古い**痕跡のクラスタ（cos ≥ θ_related、≤8）を作り、LLM が **keep** か **replace** を判定。要旨は最強成員の安定度＋他の「生きた証拠」を継承。無関係な出力は作話として拒否。 |

層・カウンタ・リング・保守呼び出しは存在しません。どの操作もストアの線形走査 1 回以下（id・本文の検索は O(1)、淘汰と近傍の選抜は全件ソートではなく部分選択）で、経過時間には依存しません。3000 仮想年のシミュレーションがテストに含まれています。

## クイックスタート

必要なもの: [Rust](https://rustup.rs) と CMake・libclang を含む C/C++ ツールチェーン（Windows では C++ ワークロード入りの Visual Studio Build Tools に両方入っています）。CUDA Toolkit があれば埋め込みモデルは自動で GPU に載ります。

```bash
mkdir secrets && copy .env.example secrets\.env       # LLM は既定で Claude Code 経由の Claude（キー不要、下記参照）
# embeddinggemma-300m-qat-Q4_0.gguf を ./model に置く（https://ai.google.dev/gemma/docs/embeddinggemma）
start.bat            # Windows  → http://localhost:8501
./start.sh           # macOS / Linux
```

CLI（`build.bat` / `build.sh` の後）:

```bash
target/release/ltm run --dream 5 --inspect     # シード再生 → LLM 5 回の夢 → ストアをダンプ
target/release/ltm run --no-seed --say "京都に住んでいます"
cargo test                                     # 決定論的テスト一式（モデル・キー不要、Dart との言語間一致テスト含む）
cargo test --release --test longevity -- --ignored   # 3000 仮想年シミュレーション（約 15 秒）
```

ビルド・llama.cpp の CPU スレッド・ベンチの並列数は、いずれも CPU コアの 8 割以下に抑えています。

## ライブラリとして使う

`Store`（4 操作。`SqliteStore` / `MemoryStore`）、`Embedder`（`LlamaEmbedder`、OpenAI 互換 `/embeddings` 用の `HttpEmbedder`）、`Adjudicator`（任意のクロージャ）の 3 つの継ぎ目だけを差し替えます。`app` 機能を外すと依存は `serde_json` と `jiff` のみです（`default-features = false, features = ["sqlite"]`）。使用例は [README.md](README.md#using-the-library) を参照。

## LLM: Claude Code 経由の Claude

既定のプロバイダは `claude-code` です。Claude Code の CLI（`claude -p`）で Claude Haiku 5.5 を呼び、アプリ側に API キーは置きません。応答は (モデル, システムプロンプト, プロンプト) のハッシュをキーに `LLM_CACHE`（既定 `data/llm_cache.json`）へ保存され、同じシナリオの再実行では一度だけ生成した同じデータを再生します。CLI が答えられないプロンプト（未インストール・未ログイン）は `<cache>.pending.json` に書き出されるので、任意の Claude クライアントで回答してキャッシュへ統合します。`ltm run --epoch <unix秒>` は仮想時計と記憶 id を固定し、再実行でまったく同じプロンプトを送ります。`bench/llm_cache.json` には `ltm run --epoch 1780000000 --dream 5` と `ltm bench supersede --run` の Haiku 応答が入っています（`LLM_CACHE=bench/llm_cache.json`）。`deepseek`（OpenAI 互換）と `gemini` も選べます。

## 1 ターンの流れ

```
recall(発話) → LLM（システムプロンプト + 記憶パック + save_memory / delete_memory ツール） → cite(応答)
```

LLM は長期的に役立つ事実を代名詞なし・絶対日付の命題として保存します（任意で `salience` 1–10、必須の `cue` = この事実が後に問われる質問文）。夢フェーズはこの `cue` で過去を検索するため、更新は字面が遠くても自分が置き換える版に到達できます。保存が無かったターンは抽出呼び出しが 1 回だけ補います。

## ストレージ

1 テーブル `memory(id, text, created_at, tz, last_recall, stability, consolidated, model_id, vector, cue)` ＋ UI 用のターンログ（Python 版とスキーマ互換で、既存の DB をそのまま開けます）。全件を RAM に持ち（1 万件 × 768 次元でプロセス全体約 80 MB）、SQLite は永続化のみ。エンジンの各操作は 1 トランザクションで、コミット成功後にだけ RAM に反映します。毎回の夢の前に 8 世代のスナップショットを回します。

## パラメータ

19 個のエンジンパラメータは `engram::Config` にあり UI から編集できます（[`SPEC.md`](SPEC.md) §6）。`cosine_floor = 0.4`・`theta_related = 0.55`・`gist_min_cosine = 0.5` は埋め込みモデル依存で、EmbeddingGemma 向けに設定済み（別モデルは `ltm bench generality` で較正）。

## ベンチマーク

```bash
ltm bench cache                 # bench/corpus を設定中のモデルで 1 回だけ埋め込む
ltm bench sweep && ltm bench sweep --report     # 忘却: 155 実行（ENGRAM vs FIFO / LRU / random / none）
ltm bench supersede --analyze   # 上書き忘却: cue による捕捉率（--run で実 LLM の夢）
ltm bench generality --dims 768,512,256,128     # 別モデル・別次元
ltm bench perf --n 10000        # 全操作のレイテンシとメモリ
```

## Rust 1.0.0 と置き換え前の Python 1.0.1 の比較

同一マシン（8 コア 16 スレッド、RTX 5070 Ti）・同一入力での計測です。

| | Python 1.0.1 | Rust 1.0.0 |
|---|---|---|
| remember（1 万件 × 768 次元、充填中平均 / 容量上限時） | 10.8 ms / 24.8 ms | **2.0 ms / 4.3 ms** |
| recall / クラスタのプレビュー / dream(5)（1 万件） | 5.3 / 82.5 / 246 ms | **3.8 / 39.0 / 約 240 ms**（dream の大半は夢前の DB スナップショット） |
| 1 万件の読込 | 162 ms | **73 ms** |
| 1 万件時のプロセスメモリ（エンジン分） | 144 MB（113 MB） | **85 MB（76 MB）** |
| 埋め込み 1 文あたり | 21 ms（CPU） | 22 ms（CPU 12 スレッド）/ **1.3 ms（GPU）** |
| 忘却ベンチ 1 実行 / 3000 仮想年テスト | 12 秒 / 数分 | **2.7 秒 / 15 秒** |
| 忘却スイープ 155 実行（同一ベクトル） | — | 2 分 47 秒。構成ごとの中央値はすべて Python と一致、各実行の指標差は 0.0001 以内（小数の丸めのみ） |
| シード付き 220 ステップのワークロード | float32 の BLAS 内積 | 判断はすべて同一、数値差は相対 8.3e-8 以内。**Dart 版とはビット一致** |
| 埋め込みベクトル（llama-cpp-python 比） | — | コサイン ≥ 0.9993（中央値 0.99993） |
| 更新 120 ペアの cue 捕捉（θ = 0.55） | 91 / 120 | 92 / 120 |
| 上書き忘却（実 LLM の夢） | stale 0.033（DeepSeek） | stale 0.033（Claude Haiku 5.5、応答は `bench/llm_cache.json` に記録） |

## ライセンス

[MIT](LICENSE)
