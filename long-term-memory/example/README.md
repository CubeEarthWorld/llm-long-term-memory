# Examples

## `main.dart` — zero-dependency quick start

Bundled `InMemoryStore` + a toy hash embedder; shows save → retrieve →
prompt building → maintain → dream end to end.

```bash
dart run example/main.dart
```

## `sqlite_adapter/` — production-style database adapter

A complete `MemoryStore` over [`package:sqlite3`](https://pub.dev/packages/sqlite3)
with WAL, real transactions for dream adjudications, a rotating snapshot
ring (`backup()`) and checkpoint+vacuum (`compact()`).

```bash
cd example/sqlite_adapter
dart pub get
dart test
```

Use it from your app (copy the file, or depend on the folder):

```dart
final store = SqliteMemoryStore('/path/to/memory.db');
final memory = EngramMemory(store: store, embedder: myEmbedder);
await memory.initialize();
```

On Flutter, add [`sqlite3_flutter_libs`](https://pub.dev/packages/sqlite3_flutter_libs)
to bundle the native library; the adapter code works unchanged. The same
file is the recommended template for drift / Isar / Hive / ObjectBox
adapters — implement the `MemoryStore` primitives, then override the bulk
methods your backend can do in one statement.

## Wiring real models

### firebase_ai (Gemini embeddings + Gemini dream LLM)

```dart
import 'package:firebase_ai/firebase_ai.dart';

final embeddingModel =
    FirebaseAI.googleAI().generativeModel(model: 'gemini-embedding-001');

final embedder = CallbackEmbedder(
  modelId: 'gemini-embedding-001',
  dimension: 768,
  onEmbedDocuments: (texts) async {
    final res = await embeddingModel.batchEmbedContents([
      for (final t in texts)
        EmbedContentRequest(Content.text(t),
            taskType: TaskType.retrievalDocument),
    ]);
    return [for (final e in res.embeddings) Float32List.fromList(e.values)];
  },
  onEmbedQueries: (texts) async {
    final res = await embeddingModel.batchEmbedContents([
      for (final t in texts)
        EmbedContentRequest(Content.text(t),
            taskType: TaskType.retrievalQuery),
    ]);
    return [for (final e in res.embeddings) Float32List.fromList(e.values)];
  },
);

// Dream adjudication with a Gemini chat model in JSON mode:
final chat = FirebaseAI.googleAI().generativeModel(
  model: 'gemini-3.5-flash',
  generationConfig: GenerationConfig(responseMimeType: 'application/json'),
);
await memory.dream(adjudicate: (request) async {
  final res = await chat.generateContent([Content.text(request.buildPrompt())]);
  return DreamDecision.parseJson(res.text);
});
```

(API names follow the firebase_ai docs; check your installed version.)

### llamadart (fully local: GGUF embedding + GGUF LLM)

```dart
final embedLlama = Llama(modelPath: 'embeddinggemma-300m.gguf', embedding: true);
final embedder = CallbackEmbedder(
  modelId: 'embeddinggemma-300m',
  dimension: 768,
  // EmbeddingGemma's asymmetric prompts:
  onEmbedDocuments: (texts) async => [
    for (final t in texts)
      Float32List.fromList(await embedLlama.embed('title: none | text: $t')),
  ],
  onEmbedQueries: (texts) async => [
    for (final t in texts)
      Float32List.fromList(await embedLlama.embed('task: search result | query: $t')),
  ],
);

final chatLlama = Llama(modelPath: 'qwen2.5-3b-instruct.gguf');
await memory.dream(adjudicate: (request) async {
  final raw = await chatLlama.generate(request.buildPrompt());
  return DreamDecision.parseJson(raw);
});
```

### Conversation loop (any tool-calling LLM)

```dart
final pack = await memory.retrieve(userText);
final response = await yourLlm.chat(
  system: EngramPrompts.conversationSystemPromptJa,
  user: EngramPrompts.buildUserMessage(
    currentTime: memory.nowLocal(),
    memoryPack: pack.packText,
    userText: userText,
  ),
  tools: [EngramPrompts.saveMemoryToolSpec, EngramPrompts.deleteMemoryToolSpec],
);
for (final call in response.toolCalls) {
  switch (call.name) {
    case 'save_memory':
      await memory.saveMemory(call.args['text'] as String);
    case 'delete_memory':
      await memory.deleteMemory(call.args['id'] as String);
  }
}
await memory.maintain();
```
