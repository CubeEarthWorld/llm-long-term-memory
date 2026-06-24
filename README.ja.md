# LLM Long-Term Memory

LLM に長期記憶を持たせるレイヤー。内部エンジンは **ENGRAM v1.1** 仕様（[`ENGRAM_spec_v1_1.md`](ENGRAM_spec_v1_1.md)）の完全実装です。常時依存はテキスト埋め込みモデル（参照実装: EmbeddingGemma）と単一ファイルDB（SQLite）のみ。生成LLMの使用は **書込み・読出し後の応答・夢（統合）** の3点に限定されます。

> 生成は言語化の瞬間だけ。判断はすべて距離。忘却はすべて算術。破壊はすべて夢の中。

- **テキストが正本、ベクトルは索引。** 記憶は短い自己完結テキスト（≤170字の1命題）。`vec` は導出物（キャッシュ）で再生成可能。埋め込みモデルが滅んでも記憶は死なない。
- **3層**（MRL次元切詰め＝忘却の解像度）: **L1 エピソード**（768d f32, τ=7日）/ **L2 意味**（256d int8, τ=90日）/ **L3 スキーマ**（128d int8, τ=3年）。
- **活性** `A = mass·2^(−Δt/τ)` がコサインスコアを再重み付け。同一性はコサイン距離の閾値のみ。DB全体 **<10MB**、検索は全件総当たりコサイン（`<1ms`、ベクトルDB/FAISS 依存ゼロ）。
- DBは**自己記述**: `spec` テーブルに本仕様全文を平文同梱。

製品名・ファイル構成・DB名は維持し、アルゴリズム・スキーマ・パラメータ・UIの中身のみ ENGRAM 化しています。

---

## 3層と活性・スコア

| 層 | 容量 | ベクトル（MRL） | 半減期 τ |
|----|------|----------------|---------|
| **L1 エピソード** | 1000 | 768d f32 | 7日 |
| **L2 意味** | 3000 | 256d int8 | 90日 |
| **L3 スキーマ** | 6000 | 128d int8 | 3年 |

上記のベクトル次元は EmbeddingGemma の**既定値**であり、固定依存ではありません。埋め込み全次元は `EMBEDDING_DIM`（または `GlobalConfig.dim_full`）、層ごとの MRL 切詰め長は `dim1` / `dim2` / `dim3`、MMR・夢クラスタリングの共有部分空間は `dim_coarse` で定義できます。`dim_full ≥ dim1 ≥ dim2 ≥ dim3 ≥ dim_coarse > 0` を満たす任意の組み合わせが受理されます（設定読込時に検証）。これにより、使用する重み付けモデル・方針・需要に合わせて、コードを変更せず次元数を定義できます。

本文は全層で無劣化。降格で劣化するのは検索キー（ベクトル）のみ。会話中は追記のみ（非破壊）。

```text
A(now)   = mass × 2^( −max(0, now − last_access) / τ_tier )
想起更新 : mass ← A(now);  if now − last_bonus_at ≥ 3600: mass ← min(mass+1, 64)
A_abs    = ln(1 + A) / ln(1 + 64)
score(m) = max(0, cos(query, m)) × (α + (1−α) × A_abs),  α = 0.35
注入     : MMR（λ=0.3）で5件、ヘッダ込み ≤1024字
```

## 同一性の閾値（文書—文書）

| cos | 判定 | 動作 |
|-----|------|------|
| ≥ 0.97 | 同一命題の更新 | 旧に墓標、新を挿入（再固定化） |
| 0.85–0.97 | 競合 | 両保持＋`conflict` キュー（夢で裁定） |
| < 0.85 | 新規 | 挿入、mass=1 |

完全一致テキストは新規行を作らず想起扱い（mass+1）。機械移動（LLM不要）: 昇格 A≥16、降格 A<4、淘汰は L3 溢れを A 昇順で物理削除。**不死記憶の非存在**: mass≤64 より沈黙後 6τ で A<1。

## DREAM（オフライン・破壊的操作はここだけ）

クラスタ化→LLM審理（統合/分割/変更なし）→統合元を物理削除。1審理=1トランザクション。内容アドレス指紋＋`dream_log` で空転防止。スナップショット8世代で保護、プロンプト契約で作話を抑止。

## 評価ベンチ

```bash
python eval/run_eval.py   # モック埋め込み/LLM＋仮想時計。APIキー不要
```
活性減衰 / 想起ボーナス＋不応期 / 同一性閾値 / 層降格・昇格・淘汰 / 不死非存在 / 夢の統合 を決定論的に検証。

## 起動

```bash
mkdir -p secrets && cp .env.example secrets/.env   # DEEPSEEK_API_KEY / HF_TOKEN を設定
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
# Windows: start.bat  /  macOS・Linux: start.sh  → http://localhost:8501
python cli.py --seed --dream 3 --inspect
```

会話ターンでは、モデルが応答しつつ `save_memory(text)` / `delete_memory(id)` ツールで保存・削除を判断します。既定TZは `Asia/Tokyo`（`MEMORY_TZ` で上書き）。

詳細: [README.md](README.md) / [docs.html](docs.html) / [ENGRAM_spec_v1_1.md](ENGRAM_spec_v1_1.md)

## ライセンス

[MIT](LICENSE)
