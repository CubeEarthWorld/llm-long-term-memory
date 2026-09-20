# LLM Long-Term Memory

LLM に長期記憶を与えるレイヤー。**ENGRAM v2**（[`SPEC.md`](SPEC.md)）を実装しています。人間の記憶の原理から導いた**痕跡（trace）モデル**で、**ローカルの埋め込みモデル**（EmbeddingGemma の GGUF を llama.cpp で実行）と**単一ファイルの SQLite** だけで動きます。LLM が生成するのは 書込み・利用（引用）・オフラインの**夢**（統合）の 3 点だけです。

> 生成は言語化の瞬間だけ。判断はすべて距離。忘却はすべて算術。統合はすべて夢の中。

同じアルゴリズムを Dart パッケージ（[`../long-term-memory`](../long-term-memory)）としても提供しています。共通のシナリオを両言語で再生して同一トレースを要求する言語間一致テストがあります。

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
| `remember(text, salience, cue)` | 同一テキストはリハーサル。それ以外は挿入し、**決して上書きしない**。`cue` が近傍（cos ≥ θ_related）に届く痕跡は*不安定*に生まれる。既存の痕跡には一切手を触れない。容量超過時は猶予期間（3 日）外で strength 最小の痕跡を忘却。 |
| `recall(query)` | 複数手がかりのコサイン → `score = a·(α + (1−α)·R)` → 絶対・相対閾値 → MMR → `[unix tz] text 《id》` を ≤1024 字で注入。注入は「露出」なので半分だけ強化。 |
| `cite(reply)` | LLM が引用した《id》の記憶を「使用」として完全に強化。 |
| `forget(id)` | id 指定の物理削除。 |
| `dream(budget)` | 不安定な痕跡（安定度順）を種に、その `cue` が再活性化した**より古い**痕跡のクラスタ（cos ≥ θ_related、≤8）を作り、LLM が **keep** か **replace**（更新される古い記憶の id ＋ 要旨テキスト）を判定。要旨は最強成員の安定度＋他の「生きた証拠」を継承。無関係な出力は作話として拒否。整理済みのストアでは LLM を呼ばない。 |

層・カウンタ・リング・保守呼び出しは存在しません。全状態が有界なので演算コストは経過時間に依存せず、3000 仮想年のシミュレーションがテストに含まれています。

## クイックスタート

```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
mkdir secrets && copy .env.example secrets\.env       # DEEPSEEK_API_KEY（または GEMINI_API_KEY）を記入
# embeddinggemma-300m-qat-Q4_0.gguf を ./model に置く（https://ai.google.dev/gemma/docs/embeddinggemma）
start.bat            # Windows  → http://localhost:8501
./start.sh           # macOS / Linux
```

CLI:

```bash
python cli.py --seed --dream 5 --inspect     # シード再生 → LLM 5 回の夢 → ストアをダンプ
python cli.py --say "京都に住んでいます"
python -m pytest                             # 決定論的テスト一式（モデル・キー不要、Dart との言語間一致テスト含む）
python -m pytest -m slow                     # 3000 仮想年シミュレーション（数分）
```

## 1 ターンの流れ

```
recall(発話) → LLM（システムプロンプト + 記憶パック + save_memory / delete_memory ツール） → cite(応答)
```

LLM は長期的に役立つ事実を代名詞なし・絶対日付の命題として保存します（任意で `salience` 1–10 = 情動的重み）。保存には `cue` が必須で、これは「この事実が後に問われるであろう質問文」です。夢フェーズはこの `cue` で過去を検索するため、更新と被更新が字面として遠くても、更新は自分が置き換えるべき版に到達できます。いま提示された事実は再保存せず、使った記憶の《id》を応答末尾に引用します。保存が無かったターンは抽出呼び出しが 1 回だけ同じ経路で補います。

## ストレージ

1 テーブル `memory(id, text, created_at, tz, last_recall, stability, consolidated, model_id, vector, cue)` ＋ UI 用のターンログ。全件を RAM に持ち（1 万件 × 768 次元で約 35 MB）、SQLite は永続化のみ。毎回の夢の前に 8 世代のスナップショットを回します。埋め込みモデルを替えると本文から全件を再埋め込みします（テキストが正本、ベクトルは索引）。

## パラメータ

19 個のエンジンパラメータは `LongTermMemoryConfig`（`config.py`）にあり UI から編集できます（[`SPEC.md`](SPEC.md) §6）。`cosine_floor = 0.4`・`theta_related = 0.55`・`gist_min_cosine = 0.5` は埋め込みモデルのコサイン分布に依存し、EmbeddingGemma 向けに設定済みです。

## 構成

```
├── memory/engine.py      # ENGRAM v2 エンジン（remember / recall / cite / forget / dream）
├── memory/model.py       # 痕跡 Memory
├── memory/util.py        # id・本文の清浄化・手がかり分割・任意の年のローカル時刻
├── core/embedding.py     # EmbeddingGemma GGUF（llama.cpp、オフライン）
├── core/storage.py       # SQLite 基盤 + スナップショットリング
├── core/llm_client.py    # DeepSeek / Gemini: 会話（ツール + 引用）・抽出・夢
├── core/turn.py          # ターン実行（+ core/metrics.py）
├── core/session.py       # アプリセッション: 組立・ターンログ・シード再生・夢
├── core/seed.py          # 既定シードシナリオ + シード CSV
├── server.py / jobs.py / frontend  # FastAPI ルート・ジョブ状態・ビルド不要の React UI
├── cli.py                # ヘッドレス実行
├── tests/                # pytest + 決定論的フェイク（一致テストは ../long-term-memory/test/conformance を読む）
└── SPEC.md               # 仕様書
```

## ライセンス

[MIT](LICENSE)
