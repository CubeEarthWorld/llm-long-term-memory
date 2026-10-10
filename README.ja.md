# LLM Long-Term Memory

LLM に長期記憶を与えるエンジンです。**ENGRAM v2.1**（[`SPEC.md`](SPEC.md)）を実装しています。人間の記憶の原理から導いた**痕跡（trace）モデル**です。パッケージは Rust のクレート `engram` で、中身はアルゴリズムだけです。埋め込みモデル・LLM・データベースは 4 つの小さなトレイトを通して利用者が持ち込みます。クレート自体は通信機能も設定ファイルも持たず、依存は `serde_json`・`jiff`・`ulid` の 3 つだけです。

English: [README.md](README.md)　中文: [README.zh.md](README.zh.md)

---

## モデルの全体像

記憶は**痕跡**です。持つのは 2 つの数 — 最後に想起した時刻と安定度（半減期）— と 1 つのフラグ（`consolidated`）だけです。

```text
R(now)   = 2^(−max(0, now − last_recall) / stability)          想起可能性 ∈ [0,1]
strength = stability · R                                       将来の想起可能性の総量
a        = max(0, (cos − cosine_floor) / (1 − cosine_floor))   手がかりの活性化
想起      : stability ← min(stability · (1 + gain·a·(1−R)), S_max);  last_recall ← now
新規      : stability = clamp(S0 · salience, 1 s, S_max)
```

| 動詞 | 動作 |
|---|---|
| `remember(text, salience, cue)` | 同一テキストはリハーサル。それ以外は挿入し、**決して上書きしない**。`cue` がより古い近傍（cos ≥ θ_related）に届く痕跡は*不安定*に生まれる。既存の痕跡には触れない。容量超過時は猶予期間（3 日）外で strength 最小の痕跡を忘却。 |
| `recall(query)` | 複数手がかりのコサイン → `score = a·(α + (1−α)·R)` → 絶対・相対閾値 → MMR → `[unix tz] text 《id》` を ≤1024 字で注入。注入は「露出」なので半分だけ強化。`peek(query)` は副作用なしの同じ検索。 |
| `cite(reply)` | LLM が引用した《id》の記憶を「使用」として完全に強化。 |
| `forget(id)` | id 指定の物理削除。 |
| `dream(adjudicator, budget)` | 不安定な痕跡（先入れ先出し、≤ 8·budget）を種に、その `cue` が再活性化した**より古い**痕跡のクラスタ（cos ≥ θ_related、≤8）を作り、LLM が **keep** か **replace** を判定。要旨は最強成員の安定度＋他の「生きた証拠」を継承。無関係な出力は作話として拒否。 |

層・カウンタ・リング・保守呼び出しは存在しません。どの操作もストアの線形走査 1 回以下で、経過時間には依存しません。3000 仮想年のシミュレーションがテストに含まれています。

## ライブラリの使い方

```toml
[dependencies]
llm-long-term-memory = { git = "https://github.com/CubeEarthWorld/llm-long-term-memory" }
```

```rust
use engram::{Config, DreamDecision, DreamRequest, Engram, MemoryStore};

let mut memory = Engram::open(Box::new(MemoryStore::default()), Box::new(my_embedder), Config::default(), "Asia/Tokyo")?;
memory.remember("ユーザーは2026年4月に大阪へ引っ越した", 1.0, "ユーザーはどこに住んでいるか？")?;
let pack = memory.recall("今どこに住んでる？")?.pack_text;          // プロンプトに注入
memory.cite(&llm_reply)?;                                           // 使われた記憶を強化
memory.dream(&mut |req: &DreamRequest| {                            // JSON を返せる任意の LLM
    DreamDecision::parse(&my_llm(engram::DREAM_SYSTEM_PROMPT, &req.prompt())).ok_or("unparsable".into())
}, None)?;
```

差し替え口:

| トレイト / 型 | 利用者が用意するもの | クレート内にあるもの |
|---|---|---|
| `Embedder` | テキスト → ベクトル（`model_id`、`dimension`、文書用・検索用プロンプト） | — |
| `Store` | 永続化: `load_all`、原子的な `write`、`clear`、任意で `backup` | `MemoryStore`（RAM のみ） |
| `Adjudicator` | 夢の判定。任意の `FnMut(&DreamRequest) -> Result<DreamDecision>` | 寛容な JSON 解析 `DreamDecision::parse` |
| `Clock` | Unix 秒 | `system_clock()`、`ManualClock` |
| `IdGen` | 痕跡の id（一意で、作成順に並ぶこと） | `ulid_ids()` — モノトニック ULID（既定） |

埋め込みに失敗する Embedder はオフラインとして扱います。書き込みは未索引のまま保持し、応答が戻った時点で索引化します。モデルを切り替える（`model_id` が変わる）と、全痕跡を本文から埋め込み直します。

## ツール（パッケージには含まれません）

`bench/` は `engram` を利用する別の非公開クレートで、実際の組み込み例であり、ベンチマークの実行環境です。SQLite の `Store`（夢の前に 8 世代のスナップショットを保存）、GGUF モデル用の llama.cpp `Embedder`（既定は EmbeddingGemma。CUDA があれば GPU）、LLM としての Claude Code 経由の Claude、そして `ltm` コマンドを提供します。

必要なもの: [Rust](https://rustup.rs) と CMake・libclang を含む C/C++ ツールチェーン（Windows では C++ ワークロード入りの Visual Studio Build Tools）。ビルドは CPU コアの 8 割までしか使いません。

```bash
mkdir secrets && copy .env.example secrets\.env       # ツールの設定（API キー不要: Claude Code が自分でサインイン）
# embeddinggemma-300m-qat-Q4_0.gguf を ./model に置く（https://ai.google.dev/gemma/docs/embeddinggemma）
build.bat                                             # または ./build.sh
target/release/ltm run --dream 5 --inspect            # シードシナリオを再生し、LLM 5 回で夢を見て、ストアを表示
target/release/ltm run --no-seed --say "京都に住んでいます"
```

ランナーの 1 ターンは `recall(発話) → LLM（記憶パック + save_memory / delete_memory）→ cite(応答)` です。LLM は代名詞を含まず絶対日付で書いた命題を、任意の `salience` と必須の `cue`（その事実が後で問われるであろう問い）付きで保存します。夢はその cue で過去を引き直すので、文面が遠くても更新は自分が置き換える版に届きます。

LLM の回答はすべて（モデル, システムプロンプト, プロンプト）のハッシュをキーに `LLM_CACHE` へ保存し、CLI が答えられなかったプロンプトは `<cache>.pending.json` に溜めます。`ltm run --epoch <unix>` は時計と id を固定するので、再実行は同じプロンプトを送ります。`bench/llm_cache.json` には `ltm run --epoch 1780000000 --dream 5` と `ltm bench supersede --run` に対する Claude Haiku 5.5 の回答が入っています。

```bash
cargo test --workspace                          # 決定的なテスト一式（モデルも LLM も不要）
cargo test --release --test longevity -- --ignored   # 3000 仮想年のシミュレーション
ltm bench cache                                 # bench/corpus を設定中のモデルで一度だけ埋め込む
ltm bench sweep && ltm bench sweep --report     # 忘却: 155 回（engram 対 FIFO / LRU / ランダム / 忘却なし）
ltm bench supersede --analyze                   # 上書き忘却: cue による捕捉（--run で LLM を使った夢）
ltm bench generality --dims 768,512,256,128     # 他のモデル・次元。モデル依存の 3 パラメータを較正
ltm bench perf --n 10000                        # 全操作の遅延とメモリ
```

## パラメータ

19 個のエンジンパラメータはすべて `engram::Config` にあります（`to_json` / `merge_json` で JSON と相互変換）。[`SPEC.md`](SPEC.md) §6 を参照してください。埋め込みモデルのコサイン分布に依存する 3 つは EmbeddingGemma 向けに設定済みです: `cosine_floor = 0.4`、`theta_related = 0.55`、`gist_min_cosine = 0.5`。

## 性能

10,000 痕跡 × 768 次元、1 台のマシン（8 コア / 16 スレッド、RTX 5070 Ti）、ツールの SQLite ストア（ディスク）経由。5 回の中央値。

| 操作 | 時間 |
|---|---|
| remember（充填中の平均 / 容量到達後、淘汰あり） | 1.1 ms / 3.4 ms |
| recall（手がかり 1 つ） | 2.4 ms |
| クラスタのプレビュー（種 40） | 25 ms |
| 夢、判定 5 回（大半は夢の前の DB スナップショット） | ~270 ms |
| 10k 痕跡の読込 | ~100 ms |
| 1 文の埋め込み: GPU / CPU | 1.3 ms / 22 ms |
| 10k 痕跡でのプロセスメモリ | 83 MB |

コサインは 16 本の SIMD レーンで float32 の積和を取り、f64 で合計しています。

## 構成

```
├── src/                    # パッケージ（クレート engram）
│   ├── engine.rs           #   remember / recall / peek / cite / forget / dream
│   ├── store.rs embed.rs dream.rs time.rs id.rs   # 差し替え口: Store, Embedder, Adjudicator, Clock, IdGen
│   └── config.rs trace.rs text.rs vector.rs json.rs
├── tests/                  # パッケージのテスト（メモリ上のストア、決定的なフェイク）
├── bench/                  # ツール（クレート ltm-bench、コマンド ltm）とベンチマークデータ
│   ├── src/                #   sqlite.rs llama.rs claude.rs、ランナー（session / turn / seed / cli）、bench/
│   └── tests/
└── SPEC.md                 # 仕様
```

## ライセンス

[MIT](LICENSE)
