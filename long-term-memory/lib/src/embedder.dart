import 'dart:async';
import 'dart:typed_data';

/// The embedding model interface — implemented by the application.
///
/// The package never loads a model itself: wrap whatever you use
/// (firebase_ai, llamadart, an ONNX runtime, a REST endpoint, ...) in this
/// interface. Asymmetric retrieval prompts are supported by having separate
/// query/document methods; if your model is symmetric, implement both with
/// the same call.
///
/// Returned vectors do **not** need to be normalised — the engine
/// L2-normalises every vector it receives, then applies Matryoshka (MRL)
/// truncation per tier. For best L2/L3 quality use an MRL-trained model
/// (e.g. EmbeddingGemma); with a non-MRL model the engine still works but
/// truncated tiers lose more precision.
abstract class Embedder {
  /// Stable identifier of the model (e.g. `'google/embeddinggemma-300m'`).
  /// Stamped on every cached vector; vectors from other models are
  /// garbage-collected during dream housekeeping, and re-embedding heals the
  /// index after a model switch.
  String get modelId;

  /// Native output dimension. Must be ≥ `EngramConfig.dim1` (768 by default).
  int get dimension;

  /// Embeds search queries (the "query" side of an asymmetric model).
  Future<List<Float32List>> embedQueries(List<String> texts);

  /// Embeds stored propositions (the "document" side).
  Future<List<Float32List>> embedDocuments(List<String> texts);
}

/// Convenience [Embedder] built from callbacks — handy when wiring an SDK
/// without declaring a class:
///
/// ```dart
/// final embedder = CallbackEmbedder(
///   modelId: 'text-embedding-004',
///   dimension: 768,
///   onEmbedQueries: (texts) => myApi.embed(texts, taskType: 'RETRIEVAL_QUERY'),
///   onEmbedDocuments: (texts) => myApi.embed(texts, taskType: 'RETRIEVAL_DOCUMENT'),
/// );
/// ```
class CallbackEmbedder implements Embedder {
  /// Creates an embedder from two callbacks. If [onEmbedQueries] is omitted,
  /// queries use [onEmbedDocuments] (symmetric model).
  CallbackEmbedder({
    required this.modelId,
    required this.dimension,
    required Future<List<Float32List>> Function(List<String> texts)
        onEmbedDocuments,
    Future<List<Float32List>> Function(List<String> texts)? onEmbedQueries,
  })  : _documents = onEmbedDocuments,
        _queries = onEmbedQueries ?? onEmbedDocuments;

  @override
  final String modelId;

  @override
  final int dimension;

  final Future<List<Float32List>> Function(List<String>) _queries;
  final Future<List<Float32List>> Function(List<String>) _documents;

  @override
  Future<List<Float32List>> embedQueries(List<String> texts) => _queries(texts);

  @override
  Future<List<Float32List>> embedDocuments(List<String> texts) =>
      _documents(texts);
}
