/// A portable long-term memory engine for LLM applications (ENGRAM v1.1).
///
/// Bring your own LLM, embedding model and database — the package contains
/// only the algorithm:
///
/// ```dart
/// final memory = EngramMemory(
///   store: InMemoryStore(),          // or your own MemoryStore adapter
///   embedder: myEmbedder,            // your Embedder implementation
///   defaultTimezone: const MemoryTimezone('Asia/Tokyo', Duration(hours: 9)),
/// );
/// await memory.initialize();
///
/// await memory.saveMemory('ユーザーは京都に住んでいる');
/// final recall = await memory.retrieve('どこに住んでいる？');
/// await memory.maintain();
/// await memory.dream(adjudicate: myLlmCallback);
/// ```
///
/// See the README for the full architecture and integration recipes.
library;

export 'src/config.dart' show EngramConfig;
export 'src/dream/adjudicator.dart'
    show
        DreamAdjudicator,
        DreamClusterRequest,
        DreamDecision,
        DreamMember,
        DreamProposal,
        relaxedJsonDecode;
export 'src/dream/prompts.dart' show EngramLocale, EngramPrompts;
export 'src/embedder.dart' show CallbackEmbedder, Embedder;
export 'src/engine/engine.dart' show EngramMemory;
export 'src/models.dart'
    show
        ConflictRecord,
        DeleteAction,
        DeleteResult,
        DreamAction,
        DreamLogRecord,
        DreamReport,
        EngramStats,
        MemoryRecord,
        MemorySnapshot,
        RecalledMemory,
        RetrieveResult,
        SaveAction,
        SaveResult,
        VecDtype,
        VectorRecord;
export 'src/store/in_memory_store.dart' show InMemoryStore;
export 'src/store/memory_store.dart' show Liveness, MemoryStore, TierEntry;
export 'src/text_utils.dart' show clusterFingerprint, shorten, splitParagraphs;
export 'src/timezone.dart'
    show MemoryTimezone, formatLocal, formatUtcOffset, parseUtcOffset;
export 'src/ulid.dart' show UlidGenerator, ulid;
export 'src/vector_math.dart'
    show
        QuantizedVector,
        cohesion,
        cosine,
        dequantizeInt8,
        dot,
        l2Normalized,
        packF32,
        quantizeInt8,
        sphericalKMeans,
        truncateNormalize,
        unpackF32;
