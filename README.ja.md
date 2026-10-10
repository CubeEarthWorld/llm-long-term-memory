# LLM Long-Term Memory

LLM に長期記憶を与えるエンジンです。**ENGRAM v2.2**（[`SPEC.md`](SPEC.md)）を実装しています。人間の記憶の原理から導いた**痕跡（trace）モデル**です。パッケージは純 Rust のクレート `engram` です。エンジンは記憶のコピーを持たず、必要なことをストレージに問い合わせるので、起動は一瞬で、どの操作も索引への問い合わせ数回で済みます。同梱の `Store` はすべてを 1 つのファイル（redb）に置き、自分で細かくなっていく類似検索の索引を持ちます。`Storage` トレイトを実装すれば、任意のデータベースやベクトル索引に置き換えられます。埋め込みモデルと LLM は利用者が持ち込みます。通信機能も設定ファイルもなく、依存は `serde_json`・`jiff`・`ulid` と、同梱ストア用の `redb` だけです。

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
| `remember(text, salience, cue)` | 同一テキストはリハーサル。それ以外は挿入し、**決して上書きしない**。`cue` がより古い近傍（cos ≥ θ_related）に届く痕跡は*不安定*に生まれる。既存の痕跡には触れない。容量超過時は猶予期間（3 日）外で strength 最小の痕跡を忘却（安定度帯ごとの索引から選び、全件の順位付けはしない）。 |
| `recall(query)` | 複数手がかりのコサイン → `score = a·(α + (1−α)·R)` → 絶対・相対閾値 → MMR → `[unix tz] text 《id》` を ≤1024 字で注入。注入は「露出」なので半分だけ強化。`peek(query)` は副作用なしの同じ検索。 |
| `cite(reply)` | LLM が引用した《id》の記憶を「使用」として完全に強化。 |
| `forget(id)` | id 指定の物理削除。 |
| `dream(adjudicator, budget)` | 不安定な痕跡（先入れ先出し、≤ 8·budget）を種に、その `cue` が再活性化した**より古い**痕跡のクラスタ（cos ≥ θ_related、≤8）を作り、LLM が **keep** か **replace** を判定。要旨は最強成員の安定度＋他の「生きた証拠」を継承。無関係な出力は作話として拒否。各報告がそのまま取り消し記録（`undo(report)`）。 |

層・カウンタ・リング・保守呼び出し・スナップショットのコピーは存在しません。起動時に何も読み込まず、全件に触れる操作もありません。どの操作もストアへの問い合わせ数回（索引付きのストアなら O(log N)）で、経過時間には依存しません。3000 仮想年のシミュレーションがテストに含まれています。

## ライブラリの使い方

```toml
[dependencies]
llm-long-term-memory = { git = "https://github.com/CubeEarthWorld/llm-long-term-memory" }
```

```rust
use engram::{Config, DreamDecision, DreamRequest, Engram, Store};

let mut memory = Engram::open(Box::new(Store::open("memory.redb".as_ref())?), Box::new(my_embedder), Config::default(), "Asia/Tokyo")?;
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
| `Storage` | 永続化と検索: `find(view)`（id・本文・不安定キュー・淘汰順など）、`count`、`near`（類似検索）、原子的な `write`、`clear` | `Store`: redb ファイル 1 つ + IVF 類似検索索引（既定の機能 `store`） |
| `Adjudicator` | 夢の判定。任意の `FnMut(&DreamRequest) -> Result<DreamDecision>` | 寛容な JSON 解析 `DreamDecision::parse` |
| `Clock` | Unix 秒 | `system_clock()`、`ManualClock` |
| `IdGen` | 痕跡の id（一意で、作成順に並ぶこと） | `ulid_ids()` — モノトニック ULID（既定） |

`Store` は件数に関係なく数ミリ秒で開きます。行と各 view は 1 つのファイルの B 木で、ベクトルは代表点（セントロイド）ごとのリストに入ります。検索はクエリに近い 8 つのリストだけを読み、リストが 512 件を超えるとそのリストだけが 2 つに分かれるので、作り直しなしに索引が細かくなっていきます。メモリに載るのは代表点と上限付きのページキャッシュだけです。1 つのモデルのリストが 2 つ以上になると検索は近似になります。別のデータベースやベクトル索引（PostgreSQL + pgvector、Qdrant、LanceDB など）を使うときは `Storage` を実装し（各 view は索引 1 本の範囲で答えられます）、`default-features = false` でビルドすれば `redb` も外れます。埋め込みに失敗する Embedder はオフラインとして扱い、書き込みは未索引のまま保持して、次の夢（または `reindex(limit)`）で新しい順に埋め込み直します。モデルを切り替えた（`model_id` が変わった）ときも同じです。

## パラメータ

19 個のエンジンパラメータはすべて `engram::Config` にあります（`to_json` / `merge_json` で JSON と相互変換）。[`SPEC.md`](SPEC.md) §6 を参照してください。埋め込みモデルのコサイン分布に依存する 3 つは EmbeddingGemma 向けに設定済みです: `cosine_floor = 0.4`、`theta_related = 0.55`、`gist_min_cosine = 0.5`。

## 性能

同梱の `Store`（ディスク）、1 台のマシン（8 コア / 16 スレッド）、CPU 版、埋め込みは事前計算。768 次元は 1 万件が 5 回の中央値、10 万件は 1 回。128 次元で 100 万件まで。

| 操作 | 1 万件 × 768 | 10 万件 × 768 | 100 万件 × 128 |
|---|---|---|---|
| remember（容量到達後、淘汰あり） | 11 ms | 15 ms | 17 ms |
| recall（手がかり 1 つ） | 9 ms | 11 ms | 11 ms |
| 夢、判定 5 回 | 36 ms | 43 ms | 56 ms |
| 起動 | 4 ms | 7 ms | 7 ms |
| プロセスメモリ | 70 MB | 0.5 GB | 1.3 GB |

起動時に何も読み込まず、夢の前にコピーもしません。操作のコストは読むリストの量で決まり、件数にほぼ比例しません。メモリは 1 GiB のページキャッシュで上限が決まります。コサインは 16 本の SIMD レーンで float32 の積和を取り、f64 で合計しています。

## 構成

```
├── src/                    # パッケージ（クレート engram）
│   ├── engine.rs           #   remember / recall / peek / cite / forget / dream
│   ├── storage.rs store.rs                        # 差し替え口 Storage と同梱の Store（redb + IVF）
│   ├── embed.rs dream.rs time.rs id.rs             # その他の差し替え口: Embedder, Adjudicator, Clock, IdGen
│   └── config.rs trace.rs text.rs vector.rs json.rs
├── tests/                  # テスト（メモリ上のストア、決定的なフェイク）
└── SPEC.md                 # 仕様
```

## ライセンス

[MIT](LICENSE)
