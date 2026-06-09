# SPECA 実行分析レポート: LLM Long-Term Memory

> 実行日時: 2026-06-09 16:07-16:41 JST
> 対象プロジェクト: `C:\Users\mosim\OneDrive\Desktop\Making\LLM_Long_Memory`
> SPECAバージョン: mainブランチ最新 (commit 779a0d0)

---

## 1. 実行サマリ

| フェーズ | 状態 | 所要時間 | コスト |
|----------|------|----------|--------|
| Phase 01a (Spec Discovery) | ✅ 完了 | 438s | $10.55 |
| Phase 01b (Subgraph Extraction) | ✅ 完了 | 363s | $13.03 |
| Phase 01e (Property Generation) | ❌ API障害 | - | $0.00 |
| Phase 02c-04 | ⏭️ スキップ | - | - |

**合計消費: $23.58** (予算 $50.00 の 47.2%)

**根本原因**: Phase 01a が `CubeEarthWorld/llm-long-term-memory` ではなく `CubeEarthWorld/zenist-todo` をクロール。DeepSeek V4 Flash モデルが SPEC_URLS を無視して同一Organizationの別リポジトリを解析した。

---

## 2. SPECA自体の不具合（発見・修正済み）

### 2.1 cp932エンコーディングエラー [Critical]
**ファイル**: [`api_runner.py`](C:\Users\mosim\OneDrive\Desktop\Software\speca\scripts\orchestrator\api_runner.py:693), [`queue.py`](C:\Users\mosim\OneDrive\Desktop\Software\speca\scripts\orchestrator\queue.py:44), [`watchdog.py`](C:\Users\mosim\OneDrive\Desktop\Software\speca\scripts\orchestrator\watchdog.py:547)

日本語Windows環境（既定CP932）で `encoding="utf-8"` 指定なしにファイルを開くと `UnicodeDecodeError` が発生。Phase 01b の全バッチが3回リトライ後失敗した。

**修正**: 該当の `open()` 呼び出し全てに `encoding="utf-8"` を追加。

### 2.2 クリーンアップ時のPermissionError [High]
**ファイル**: [`resume.py`](C:\Users\mosim\OneDrive\Desktop\Software\speca\scripts\orchestrator\resume.py:212)

`cleanup_incomplete_batches()` が `shutil.rmtree()` の `PermissionError` を捕捉しておらず、古いバッチディレクトリ削除時にパイプライン全体がクラッシュ。Windows ではロックされたファイルが原因。

**修正**: `try/except PermissionError` で囲み、警告を出力して続行するよう変更。

### 2.3 API Keyの末尾スペース [Medium]
**ファイル**: [`api_runner.py`](C:\Users\mosim\OneDrive\Desktop\Software\speca\scripts\orchestrator\api_runner.py:391)

`os.environ.get(API_KEY_ENV, "")` の戻り値を `.strip()` していないため、cmd.exe の `set` コマンドで設定された環境変数の末尾スペースがそのままBearerトークンに混入。HTTPリクエストが `Illegal header value` で失敗。

**修正**: `.strip()` を追加。

### 2.4 非ClaudeモデルでのSpec Discoveryの精度問題 [High]
DeepSeek V4 Flash（`api` ランタイム）使用時、Phase 01a が指定された SPEC_URLS を無視し、GitHub同一Organizationの別リポジトリをクロールする。プロンプト追従性がClaudeに比べて著しく低い。

---

## 3. LLM_Long_Memory プロジェクトの問題点・懸念点

### 3.1 APIキーの平文バックアップ [Critical]
**ファイル**: [`secrets/.env.backup`](C:\Users\mosim\OneDrive\Desktop\Making\LLM_Long_Memory\secrets\.env.backup)

プロジェクトルートに実際のAPIキー（DeepSeek、Gemini、HuggingFace）を含む `.env.backup` ファイルが存在。ファイルには `⚠️ Rotate these after testing` とあるが、実際のキーが平文でGit管理下に置かれている可能性がある。

**推奨**: 即時ローテーション + `.env.*` をグローバル `.gitignore` に追加。

### 3.2 LLM API呼び出しのリトライ不在 [High]
**ファイル**: [`llm_client.py`](C:\Users\mosim\OneDrive\Desktop\Making\LLM_Long_Memory\core\llm_client.py:159-175)

`respond()` と `extract_memory()` にリトライロジックがない。ネットワークエラーやAPIレート制限が発生すると即座に空の結果/エラーメッセージを返す。特に `extract_memory()` は例外を握りつぶして空リストを返す（l.206-207）ため、メモリ抽出の失敗がサイレントに無視される。

```python
# llm_client.py:206-207
except Exception:
    return []  # エラー情報が完全に失われる
```

**推奨**: 指数バックオフ付きリトライ（最低3回）+ エラーログ出力。

### 3.3 マルチスレッドSQLiteアクセスの潜在的競合 [Medium]
**ファイル**: [`storage.py`](C:\Users\mosim\OneDrive\Desktop\Making\LLM_Long_Memory\core\storage.py:41), [`server.py`](C:\Users\mosim\OneDrive\Desktop\Making\LLM_Long_Memory\server.py:49)

`check_same_thread=False` + WALモードで動作しているが、`server.py` の `run_job()` が `threading.Lock` でシリアライズしているため実質的には安全。ただし `LOCK` は `run_job` の内部でのみ取得され、GET系エンドポイント（`/api/state`, `/api/db` 等）は `LOCK` を取得せずに `ENGINE` にアクセスする。

**懸念**: GETエンドポイントがジョブ実行中の `ENGINE["e"]` の部分更新を読む可能性。現状はPythonのGILとdict代入のアトミック性に依存。

### 3.4 `_gather_candidates` の日本語テキスト分割問題 [Medium]
**ファイル**: [`long_term_memory.py`](C:\Users\mosim\OneDrive\Desktop\Making\LLM_Long_Memory\memory\long_term_memory.py:445)

クエリテキストを改行でパラグラフ分割するが、日本語入力は明示的な空行なしでは単一の巨大パラグラフになる。各パラグラフを独立に埋め込み検索するため、長文クエリでは多様な検索が行われず、再現率が低下する。

**推奨**: 文分割（`。`区切り）または固定長チャンク分割を併用。

### 3.5 データベースパスのハードコード [Low]
**ファイル**: [`engine.py`](C:\Users\mosim\OneDrive\Desktop\Making\LLM_Long_Memory\core\engine.py:44-47)

`DATA_DIR` が `./data` にハードコードされており、起動時以外に変更不可。複数インスタンス実行時にDBが衝突する。

### 3.6 巨大な埋め込みモデルのメモリ消費 [Low]
**ファイル**: [`embedding.py`](C:\Users\mosim\OneDrive\Desktop\Making\LLM_Long_Memory\core\embedding.py)

`google/embeddinggemma-300m` は起動時にGPUがない場合はCPUメモリに約1.2GBを消費。低スペック環境ではスワップを引き起こす可能性がある。モデルロード失敗時のエラーハンドリングはあるが、実用的なフォールバック（より小さいモデル等）はない。

### 3.7 `server.py` の `threading.Thread` 未管理 [Info]
**ファイル**: [`server.py`](C:\Users\mosim\OneDrive\Desktop\Making\LLM_Long_Memory\server.py:211)

`run_job()` が毎回新しいdaemonスレッドを生成するが、ジョブのタイムアウトやキャンセル機構がない。長時間実行中のジョブ（dream等）を中断する手段がない。

### 3.8 フロントエンドのキャッシュ無効化（開発用設定の本番残留） [Info]
**ファイル**: [`server.py`](C:\Users\mosim\OneDrive\Desktop\Making\LLM_Long_Memory\server.py:53-64)

`NoCacheStaticFiles` が全てのHTML/JS/CSSのキャッシュを無効化している。開発中の利便性のためだが、本番配布時にはパフォーマンスに悪影響。

---

## 4. 総評

### SPECAについて
SPECAはブロックチェーンプロトコルの形式的セキュリティ監査には極めて有効だが、汎用Pythonアプリケーションのバグ検出には以下の理由で適さない：

1. 仕様書（EIP等）を前提としたパイプライン設計
2. `api` ランタイム（OpenAI互換）でのプロンプト追従性の低さ
3. 1回のパイプライン実行で $50 以上のAPI費用が発生

### LLM_Long_Memoryについて
プロトタイプとしての品質は良好。数学的に堅牢な忘却モデル（FSRS系 power-law retrievability）と明確なコード構造を持つ。実運用に向けては、API呼び出しの耐障害性とシークレット管理が最重要課題。

### 修正したspecaのバグ
| # | ファイル | 問題 | 重要度 |
|---|----------|------|--------|
| 1 | `api_runner.py:693,711,773,779` | cp932エンコーディング未指定 | Critical |
| 2 | `queue.py:44` | cp932エンコーディング未指定 | Critical |
| 3 | `watchdog.py:547` | cp932エンコーディング未指定 | Critical |
| 4 | `resume.py:212-218` | PermissionError未処理 | High |
| 5 | `api_runner.py:391` | API key末尾スペース未strip | Medium |
