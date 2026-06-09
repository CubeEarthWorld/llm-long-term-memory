# LLM Long-Term Memory

LLM に長期記憶を持たせるための軽量な仕組み。LLM を呼ぶのは**書き込み（抽出）時**と
**Dreaming（記憶統合）時**のみ。想起・スコアリング・クラスタリングはローカル Embedding
（`google/embeddinggemma-300m`）と数値計算で行います。

- 記憶は2層（SQLite）: 想起対象の **active（hot, 上限1000件）** と、忘れて引けなくなった
  **archive（cold, 上限5000件, savings 用）**。`total_cap + archive_cap` で総容量は有界（~数十MB）。
- 検索には768次元 full embedding を使用。256次元 coarse は full から都度導出し、クラスタリング・
  多様性調整・連想拡散・archive 照合に使います（archive は coarse のみ保存、full は破棄）。
- 取得 memory pack は最大1024文字。
- 各記憶は人間の記憶に倣い3軸を分離: `w`（重要度）/ `confidence`（信頼度）/ `S`（定着度=安定度）。
- 各記憶は IANA タイムゾーン付きで時刻を保持し、想起・Dreaming 時に LLM へ渡します。

## データ構造

```sql
CREATE TABLE memories (              -- active (hot)
  id TEXT PRIMARY KEY,               -- uuid7
  text TEXT, v_full BLOB,            -- 768d（coarse は都度導出）
  w REAL, provenance TEXT, confidence REAL,  -- 重要度 / 'user'|'inferred' / 信頼度
  freq REAL, stability REAL,         -- 累積アクセス回数 / 定着度 S（秒）
  accessed_at_unix REAL,             -- 最終アクセス → 減衰の基準
  updated_at_unix REAL,              -- 内容バージョン時刻 → 想起/Dream に使用
  timezone TEXT, cluster_id TEXT,
  source_ids TEXT, summary_of TEXT, dream_action TEXT, created_by_dream_id TEXT  -- Dream 由来 (lineage)
);
CREATE TABLE archive (               -- cold (savings 用、768d は破棄)
  id TEXT PRIMARY KEY,
  text TEXT, v_coarse BLOB,          -- 256d coarse のみ（再出現照合用）
  w REAL, provenance TEXT, confidence REAL,
  last_r REAL, archived_at_unix REAL, text_hash TEXT, timezone TEXT
);
CREATE TABLE clusters (
  id TEXT PRIMARY KEY,
  last_dreaming_unix REAL            -- 作成 UNIX。Dream のたびに更新
);
```

centroid / size / medoid はクラスタのメンバから都度導出します（保存しません）。

## 保持・想起・忘却

定着度 `S`（安定度）と保持率 `r`（FSRS 型）を分離。忘却は**べき乗則**で、想起成功のたびに
`S` が伸びます（忘れかけ＝`r` が低いほど大きく＝間隔/テスト効果・Jost の法則）。

```text
S0           = stab_base * (1 + kappa*w) * (labile_frac + (1-labile_frac)*confidence)
r(t)         = (1 + (now - accessed_at) / S) ^ (-forget_beta)        # 忘却曲線（べき乗則）
補強(access): S = S * (1 + stab_growth_c*(1-r)); freq += reinforce_inc; accessed_at = now
               （S と freq は長期使用で無限増大しないよう、max_stability_seconds / max_freq でクリップ）
score        = alpha*cos + beta*r + delta*w + eta*log1p(freq) + zeta*confidence  (+ 連想拡散)
想起ゲート    : gate_w_cos*cos + gate_w_r*r (+noise) >= gate_theta   # 機能的忘却
```

忘却は2段階で容量有界:

1. **active → archive**: `r` が `r_archive_floor` を（猶予期間を超えて）下回ると archive へ退避。
   想起では引けなくなる（機能的忘却）が、savings として残る。高 `w`＋高 `confidence` は
   `r_hard_floor` まで保護。容量超過時は最小 `r` から退避。
2. **archive → 永久削除**: archive が `archive_cap` を超えると最小 `last_r` から削除。
3. **savings（再出現）**: 同じ/類似テキスト（coarse cos ≥ `tau_savings`、または text_hash 一致）が
   再来すると、安定度の先取り（`* savings_gain`）で active に復元（再学習の節約）。
4. **干渉**: 想起された記憶は同クラスタの非想起競合の `S` を僅かに減衰（想起誘導性忘却）。

## Dreaming（記憶統合）

睡眠アナロジーの整理処理。オンデマンド実行で、優先度の高いクラスタを LLM が
**統合(merge) / 分割(split) / 維持(none)** します。

- 優先度（計算のみ）= `w_size*log(1+size) + w_spread*(spread/norm) + w_age*(since_dream/norm) + w_disp*dispersion`
  - `spread` = クラスタ内の `updated_at_unix` の時間幅、`since_dream` = `last_dreaming_unix` からの経過。
  - `dream_min_size` 未満、または `dream_min_interval`（クールダウン）以内のクラスタは対象外。
- LLM にはメンバの内容・重要度・**タイムゾーン付き時刻**・保持率 `r` を渡します。出力した記憶が
  入力メンバを置換します（merge=少数化・要点 gist 化、split=細分化）。陳腐化した内容は現在時刻を
  踏まえ更新（例『7月に旅行予定』→『2026年7月に旅行済み』）。矛盾は新しい時刻のものを優先。
- 統合記憶の **stability は元の最大値を超えない**（`S = max(member S)`、エピソード→意味記憶の gist 化）。`consolidation_gain` は現在未使用。
  由来 `source_ids` / `created_by_dream_id` を保持（lineage）。影響クラスタの `last_dreaming_unix` を
  更新（none を選んでも更新するので、直後の再処理を防ぐ）。

## 評価ベンチ

`eval/run_eval.py` は仮想時計＋モック埋め込み/LLM で、記憶忠実度を決定論的に検証します
（モデル DL や API キー不要）。`python eval/run_eval.py` で 8 シナリオ（間隔効果・べき乗則忘却・
想起ゲート・安定度成長・archive/savings・容量有界・3軸保護・連想拡散）を実行。

## セットアップ

1. API キーを `secrets/.env` に配置します（`secrets/` は git 対象外）。
   ```bash
   mkdir -p secrets
   cp .env.example secrets/.env
   # エディタで secrets/.env を開き、実際のキーを入力
   ```
   - **Deepseek**（既定）: `DEEPSEEK_API_KEY` を設定
   - **Gemini**（オプション）: `LLM_PROVIDER=gemini` と `GEMINI_API_KEY` を設定
   - **HuggingFace**（EmbeddingGemma 用）: `HF_TOKEN` を設定

2. 依存関係をインストールします。
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 起動

Windows: `./start.bat` ／ macOS・Linux: `./start.sh` → ブラウザで `http://localhost:8501`。
会話タブの「💤 Dream」ボタンで整理を実行できます。

```powershell
python cli.py --say "私は京都に住んでいます"
python cli.py --dream            # シード後に Dreaming を1クラスタ実行
python cli.py --dream 3 --inspect  # 上位3クラスタを整理し DB をダンプ
```

タイムゾーンの既定は `MEMORY_TZ` 環境変数（無ければ `Asia/Tokyo`）。結果は `data/results.json`。

## ライセンス

[MIT](LICENSE)

## 他言語版

- [English](README.md)
- [中文](README.zh.md)
