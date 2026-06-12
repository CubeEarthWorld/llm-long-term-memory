# long_term_memory

LLMアプリケーション向けの**長期記憶エンジン**。純粋Dartで書かれており、Flutterアプリ（全プラットフォーム）やDartサーバー/CLIから利用できます。

本リポジトリのENGRAM v1.1記憶システムの忠実な移植です。設計全体は次の4文に圧縮されます:

> 生成は言語化の瞬間だけ。判断はすべて距離。忘却はすべて算術。破壊はすべて夢の中。

**このパッケージにはアルゴリズムだけが含まれます。** LLM・Embeddingモデル・DBは含まれず、3つの小さなインターフェースを通じて外部から注入します。そのため [firebase_ai](https://pub.dev/packages/firebase_ai)、[llamadart](https://pub.dev/documentation/llamadart/latest/)、[sqlite3](https://pub.dev/documentation/sqlite3/latest/)、drift、Isar、Hive、RESTエンドポイント、オンデバイスONNXモデルなど、何とでも組み合わせられます。

English version: [README.md](README.md)

---

## 機能

- **3層記憶** — L1エピソード記憶（768次元 float32、半減期7日）、L2意味記憶（256次元 int8、90日）、L3スキーマ記憶（128次元 int8、3年）。各層は1つの埋め込みのMatryoshka（MRL）切り詰めを保持するため、低層は同じベクトルの低解像度版で済み、再埋め込みは不要です。
- **活性化減衰** — 各記憶は質量（mass）を持ち、活性 `A = mass · 2^(−Δt/τ)` が層ごとの半減期で減衰します。想起は質量を強化します（1時間の不応期付き分散効果）。忘却は純粋な算術です。
- **距離だけの同一性判定** — 書き込みは文書間コサインで判定: `≥ 0.97` は古い記憶を上書き（墓石化）、`0.85–0.97` は両方保持して競合キューへ、それ未満は新規挿入。書き込み経路にLLMは使いません。
- **関連度+新近性の検索** — スコア = `max(0, cos) · (α + (1−α)·Â)`。活性フロアにより「休眠中だが関連の高い」記憶も競争でき、MMRによる多様性選択と文字数予算付き・インジェクション安全な記憶パックを生成します。
- **夢フェーズの統合整理** — オフラインでのクラスタリング（球面k-means）+ あなたが実装するLLMコールバックによる裁定（merge / split / none）。世代上限、作話ガード、質量の引き継ぎ、1裁定=1トランザクションはエンジン側が強制します。
- **層の移動と容量管理** — 自動降格（L1→L2→L3、MRL切り詰め+int8量子化）、ホットな記憶のL1への昇格（再埋め込み）、墓石掃除、リングバッファの競合/夢ログ、ハードな行数上限。
- **時刻・タイムゾーン対応** — すべての呼び出しが明示的な64bit UNIX時刻とタイムゾーン（`IANA名+オフセット`、`'Asia/Tokyo;+09:00'` 形式で保存）を受け取れます。時計は注入可能で、時計の巻き戻しはサニタイズされ、ローカル時刻のフォーマットにtzデータベースは不要です。
- **すべて持ち込み（Bring your own）** — `Embedder`（埋め込みモデル）、`MemoryStore`（DB）、`DreamAdjudicator`（LLM）。`InMemoryStore` を同梱しているのでそのまま動作し、完全なSQLiteアダプタが [`example/sqlite_adapter`](example/sqlite_adapter) にあります。
- **プロンプトキット同梱** — 日英のシステムプロンプト、`save_memory` / `delete_memory` のツール定義、抽出フォールバックと夢統合のテンプレート、寛容なJSONパーサ（`DreamDecision.parseJson`、`EngramPrompts.parseExtractedTexts`）。
- 純粋Dart、実行時依存は `package:crypto` のみ、コード生成なし、完全に決定論的なテストスイート。

## インストール

pub.devへ公開するまでは、pathまたはgit依存で利用します:

```yaml
dependencies:
  long_term_memory:
    git:
      url: https://github.com/CubeEarthWorld/llm-long-term-memory
      path: long-term-memory
```

## クイックスタート

```dart
import 'package:long_term_memory/long_term_memory.dart';

Future<void> main() async {
  final memory = EngramMemory(
    store: InMemoryStore(),               // あなたのDBアダプタに置き換え可
    embedder: myEmbedder,                 // あなたのEmbedder実装
    defaultTimezone: const MemoryTimezone('Asia/Tokyo', Duration(hours: 9)),
  );
  await memory.initialize();

  // WRITE — 自己完結した事実を1つ保存（例: LLMツールコールから）。
  final saved = await memory.saveMemory('ユーザーは京都に住んでいる');
  print(saved.action); // SaveAction.inserted

  // READ — 次のプロンプトに注入する関連記憶を取得。
  final recall = await memory.retrieve('どこに住んでいるか覚えてる？');
  print(recall.packText); // [<unix> Asia/Tokyo;+09:00] ユーザーは京都に住んでいる　《id:...》

  // 毎ターンのメンテナンス（LLM・埋め込み不使用 — 軽量）。
  await memory.maintain();

  // あなたのLLMによるオフライン統合整理（アプリのアイドル時に実行）。
  await memory.dream(adjudicate: (request) async {
    final raw = await myLlm.generateJson(request.buildPrompt());
    return DreamDecision.parseJson(raw);
  });
}
```

依存ゼロで実行できるサンプル（トイEmbedder + InMemoryStore）は [`example/main.dart`](example/main.dart) にあります:

```bash
dart run example/main.dart
```

## チャットアプリへの組み込み方

エンジンは会話用LLMを直接呼びません。4つのプリミティブをあなたのツールコーリングループに配線します:

```
ユーザー発話
   │
   ├─► memory.retrieve(発話)              → 記憶パック
   │
   ├─► あなたのLLM呼び出し
   │     system  = EngramPrompts.conversationSystemPromptJa
   │     user    = EngramPrompts.buildUserMessage(
   │                 currentTime: memory.nowLocal(),
   │                 memoryPack: pack.packText,
   │                 userText: 発話)
   │     tools   = [EngramPrompts.saveMemoryToolSpec,
   │                EngramPrompts.deleteMemoryToolSpec]
   │
   ├─► ツールコール "save_memory"   → memory.saveMemory(args['text'])
   ├─► ツールコール "delete_memory" → memory.deleteMemory(args['id'])
   │
   └─► memory.maintain()
```

ターン中に何も保存されなかった場合は**抽出フォールバック**も使えます: `EngramPrompts.buildExtractionInstruction(...)` をLLMに送り、`EngramPrompts.parseExtractedTexts(raw)` でパースし、各文字列を `saveMemory` に通します。

## 時刻とタイムゾーンの入力

すべてのエントリポイントは、テキストと一緒に現在のUNIX時刻（秒）とタイムゾーンを受け取れます:

```dart
await memory.saveMemory(
  'ユーザーは2026-04-01にベルリンへ出張した',
  nowUnix: 1774000000,                                       // 明示的な64bit UNIX秒
  timezone: const MemoryTimezone('Europe/Berlin', Duration(hours: 2)),
);
await memory.retrieve('出張の予定', nowUnix: 1774100000);
```

- 省略時は、注入された `clock` / `defaultTimezone` → システム時計 / UTC の順でフォールバックします。
- タイムゾーンは記憶ごとに `'IANA名;+HH:MM'` で保存されます。オフセットを明示的に持つため、ローカル時刻のフォーマット（`memory.nowLocal()`、夢プロンプト）は純粋な算術で行え、tzデータベースは不要です。純粋DartはIANA名を自力で解決できないので、オフセットは呼び出し側が渡します（例: `DateTime.now().timeZoneOffset`、厳密な過去のDSTが必要なら `package:timezone`）。
- `clock` コールバック（UNIX秒を返す `int Function()`）でエンジンを完全に決定論化できます。「現在」より未来のタイムスタンプはクランプされるため、時計の巻き戻しで不死の記憶が生まれることはありません。

## 3つのインターフェース

### 1. `Embedder` — あなたの埋め込みモデル

```dart
abstract class Embedder {
  String get modelId;       // ベクトルに刻印。モデル切替時は自己修復される
  int get dimension;        // config.dim1 以上（デフォルト768）
  Future<List<Float32List>> embedQueries(List<String> texts);
  Future<List<Float32List>> embedDocuments(List<String> texts);
}
```

ベクトルの正規化は不要です（エンジン側で正規化+MRL切り詰めを行います）。L2/L3の品質のためMRL学習済みモデル（例: EmbeddingGemma）を推奨します。`CallbackEmbedder` を使えばクラス宣言なしでSDKを配線できます。

**firebase_ai の例**（Geminiの埋め込み、非対称タスクタイプ）:

```dart
final model = FirebaseAI.googleAI().generativeModel(model: 'gemini-embedding-001');
final embedder = CallbackEmbedder(
  modelId: 'gemini-embedding-001',
  dimension: 768,
  onEmbedDocuments: (texts) => /* batchEmbedContents, taskType RETRIEVAL_DOCUMENT */,
  onEmbedQueries:   (texts) => /* batchEmbedContents, taskType RETRIEVAL_QUERY */,
);
```

**llamadart の例**（ローカルGGUF埋め込みモデル）:

```dart
final llama = Llama(modelPath: 'embeddinggemma-300m.gguf', embedding: true);
final embedder = CallbackEmbedder(
  modelId: 'embeddinggemma-300m',
  dimension: 768,
  onEmbedDocuments: (texts) async => [
    for (final t in texts) Float32List.fromList(await llama.embed('title: none | text: $t')),
  ],
  onEmbedQueries: (texts) async => [
    for (final t in texts) Float32List.fromList(await llama.embed('task: search result | query: $t')),
  ],
);
```

### 2. `MemoryStore` — あなたのDB

型付きインターフェース（SQLは一切露出しません）で、記憶行・ベクトル・競合リング・夢ログ・メタデータの5グループから成ります。抽象メソッドはプリミティブのみで、バルク操作とフック（`runInTransaction`、`backup`、`compact`）にはデフォルト実装があり、効率化したい場合だけオーバーライドします。

- `InMemoryStore`（同梱）: セットアップ不要の参照実装。`toJson`/`fromJson` 付き。
- [`example/sqlite_adapter`](example/sqlite_adapter): WAL・実トランザクション・ローテーションスナップショット・vacuum を備えた完全な `package:sqlite3` アダプタ。アプリにコピーして使うか、drift/Isar/Hive/ObjectBox アダプタのテンプレートにしてください。

> **トランザクションに関する注意**: 夢の各裁定（クラスタ削除→置換挿入）は `runInTransaction` 内で実行されます。トランザクションのないストアではこの間のクラッシュで記憶が失われる可能性があります。データが重要なら `runInTransaction` と `backup` を実装してください。

### 3. `DreamAdjudicator` — あなたのLLM（夢フェーズのみ）

```dart
final reports = await memory.dream(adjudicate: (request) async {
  // request.members: id / text / gen / activation / localTime / timezone
  final raw = await myLlm.generateJson(request.buildPrompt());   // ja / en
  return DreamDecision.parseJson(raw);                            // 例外を投げない
});
```

ガードはすべてエンジン側が強制します: 置換テキストは170字に短縮、空は破棄、世代は7で上限、質量は引き継ぎ（mergeは合算、splitは均等割り、≤64）、メンバー構成が変わらず前回 `none` 判定のクラスタはスキップ、コールバックが例外を投げた場合はクラスタに手を付けず次回再試行されます。

## APIリファレンス（エンジン）

| メソッド | 役割 |
|---|---|
| `initialize()` | ストアを開き、Embedderを検証し、自己記述メタデータを書き込む。 |
| `retrieve(query, {nowUnix, timezone})` | READ: クエリのチャンク埋め込み → スコアリング → 閾値 → MMR → ≤1024字パック。注入された記憶には想起更新が入る。`RetrieveResult(packText, recalled)` を返す。 |
| `saveMemory(text, {nowUnix, timezone})` | WRITE: 検証 → 完全一致の強化 → 同一性スキャン → 挿入/上書き/競合。`SaveResult`（`inserted` / `updated` / `conflict` / `reinforced` / `rejected` / `rateLimited`）を返す。 |
| `deleteMemory(id, {hard})` | DELETE（idのみ）: 墓石化（デフォルト）または物理削除。`DeleteResult` を返す。 |
| `maintain({nowUnix})` | 毎ターンの保守: タイムスタンプ正規化、墓石掃除、容量の降格/追い出し、定期 `compact()`。LLM・埋め込み不使用。 |
| `dream(adjudicate: …, {budget, nowUnix, timezone, force})` | オフライン統合: バックアップ → ハウスキーピング → ≤budget件のLLM裁定 → 昇格 → 容量 → 圧縮。`List<DreamReport>` を返す。 |
| `stats()` / `totalRecords()` / `activationOf(id)` / `listMemories()` | 内省用。 |
| `nowUnix()` / `nowLocal({nowUnix, timezone})` | 時計ヘルパー（プロンプト用 `'2026-06-12 09:30 +09:00'`）。 |
| `reset()` | 全消去してメタデータを再インストール。 |

公開メソッドは内部で直列化されるため、`EngramMemory` インスタンスは非同期コードから安全に呼べます。夢フェーズはフル容量時にモバイルで約0.5〜2秒（純Dartのk-means、≤4000×128次元）かかります。アイドル時に実行するか、独自のストアハンドルを持つisolateで実行してください。

## 設定

ENGRAM §8の全パラメータは `EngramConfig`（const構築可、`copyWith`、JSON往復可）にあります。デフォルト値:

| グループ | パラメータ | デフォルト | 意味 |
|---|---|---|---|
| 層 | `cap1/cap2/cap3` | 1000 / 3000 / 6000 | 層ごとの行数（墓石込み） |
| | `dim1/dim2/dim3` | 768 / 256 / 128 | MRLベクトル次元（f32 / int8 / int8） |
| | `tau1/tau2/tau3` | 7日 / 90日 / 3年 | 活性の半減期 |
| 活性 | `mMax` | 64 | 質量/活性の上限 |
| | `refractorySeconds` | 3600 | 質量ボーナスの最小間隔 |
| 検索 | `alpha` | 0.35 | スコアの活性フロア |
| | `injectN` | 5 | 1回のretrieveで注入する記憶数 |
| | `mmrLambda` | 0.3 | MMR多様性ペナルティ |
| | `scoreThresholds` | [0.1, 0.2] | 段階的フィルタ（厳しい順） |
| | `budgetChars` | 1024 | パックの文字数予算 |
| 同一性 | `thetaSame` | 0.97 | ≥ → 上書き |
| | `thetaConflict` | 0.85 | 競合バンドの下限 |
| | `preciseMargin` | 0.03 | 高精度再埋め込みのマージン |
| 移動 | `thetaUp` / `thetaDown` | 16 / 4 | 昇格/降格のヒステリシス |
| テキスト | `textMax` / `textHardMax` | 170字 / 1024バイト | ソフト短縮 / ハード拒否 |
| | `genMax` | 7 | 統合世代の上限 |
| 夢 | `dreamBudget`（`dreamBudgetHard`） | 5（4096） | 1回あたりの裁定数 |
| | `clusterMin` / `clusterCohesionMin` | 3 / 0.5 | クラスタの適格条件 |
| | `dreamMaxMembers` | 64 | 1裁定あたりのメンバー数 |
| リング | `conflictCap` / `dreamLogCap` | 256 / 512 | リングバッファ容量 |
| 保守 | `tombstoneSweepPct` / `tombstoneSweepAge` | 10% / 7日 | 掃除トリガー |
| 制限 | `writeRatePerDay` | 2000 | ソフト書き込みレート |
| | `hardMemoryRows` | 16384 | ハード行数上限/ループガード |
| | `decayExpCap` | 65536 | 減衰アンダーフローガード |
| | `compactEvery` | 20 | `compact()` までのmaintain回数 |

## ストレージと性能

デフォルト容量（記憶10,000件）でストア全体は**10MB未満**: L1ベクトル3.0MB、L2 0.8MB、L3 0.8MB、テキスト+メタデータ約2MB。検索は総当たり（≤1万件×≤768次元の内積 ≈ 現代のスマホで10ms未満）で、壊れたり保守が必要になるANNインデックスはありません。

## 統合のテスト

パッケージのテストスイートがパターンを示しています: 決定論的なトークン重複 `FakeEmbedder` と `VirtualClock`（[`test/support/fakes.dart`](test/support/fakes.dart)）により、減衰・不応期・上書き/競合閾値・仮想数ヶ月にわたる夢まで、モデルもAPIキーもなしに再現できます。

```bash
dart test                                # パッケージのテスト（63件以上）
cd example/sqlite_adapter && dart test   # アダプタのテスト
```

## 設計メモと制限

- **会話用LLMは設計上アプリ側** — どのSDK・ツールコーリング規約・エージェントフレームワークとも組み合わせられます。
- **IANA tzデータベースなし** — オフセットは呼び出し側が供給し、記憶ごとに固定されます（仕様が `name;+offset` の両方を保存するのはまさにこのため）。厳密な過去のDSTが必要ならアプリ側で `package:timezone` を使ってください。
- **ソフトなテキスト長はUTF-16コードユニット数**（Dartの `String.length`）。日本語/CJKとASCIIでは文字数と一致します。
- **Embeddingモデルの切替は自己修復** — 旧モデルのベクトルは夢のハウスキーピングでGCされ、昇格時に現行モデルで再埋め込みされます。

## ライセンス

MIT — [LICENSE](LICENSE) を参照。
