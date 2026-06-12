import 'dart:async';
import 'dart:convert';

import '../models.dart';
import 'prompts.dart';

/// The dream-phase LLM callback — implemented by the application.
///
/// The engine clusters related memories and asks the callback to decide
/// merge / split / none for each cluster. Typical wiring is three lines:
///
/// ```dart
/// Future<DreamDecision> adjudicate(DreamClusterRequest request) async {
///   final raw = await myLlm.generateJson(request.buildPrompt());
///   return DreamDecision.parseJson(raw);
/// }
/// ```
///
/// If the callback throws, the engine leaves that cluster untouched (and
/// unrecorded, so the next dream retries it). All hard guards — text
/// shortening, generation cap, mass carry-over, transactional
/// delete-and-insert — are enforced by the engine, not the callback.
typedef DreamAdjudicator = FutureOr<DreamDecision> Function(
    DreamClusterRequest request);

/// One memory inside a [DreamClusterRequest].
class DreamMember {
  /// Creates a member.
  const DreamMember({
    required this.id,
    required this.text,
    required this.gen,
    required this.activation,
    required this.localTime,
    required this.timezone,
  });

  /// Memory id.
  final String id;

  /// Proposition text.
  final String text;

  /// Consolidation generation.
  final int gen;

  /// Activation at dream time (rounded).
  final double activation;

  /// Local datetime the memory was written, e.g. `'2026-05-20 14:00 +09:00'` —
  /// the reference point for resolving any relative date words in [text].
  final String localTime;

  /// Stored timezone field `'IANA;+HH:MM'`.
  final String timezone;

  /// JSON form handed to the LLM.
  Map<String, Object?> toJson() => {
        'id': id,
        'text': text,
        'gen': gen,
        'A': activation,
        'local_time': localTime,
        'timezone': timezone,
      };
}

/// One cluster of semantically close memories awaiting an LLM verdict.
class DreamClusterRequest {
  /// Creates a request.
  const DreamClusterRequest({
    required this.clusterFingerprint,
    required this.currentLocalTime,
    required this.members,
  });

  /// SHA-256 fingerprint of the membership (for logging/tracing).
  final String clusterFingerprint;

  /// Current local datetime string, e.g. `'2026-06-12 09:30 +09:00'`.
  final String currentLocalTime;

  /// The cluster members (≤ `EngramConfig.dreamMaxMembers`).
  final List<DreamMember> members;

  /// The members as pretty-printed JSON (the `listing` payload of the
  /// reference implementation).
  String membersAsJson() => const JsonEncoder.withIndent('  ')
      .convert(members.map((m) => m.toJson()).toList());

  /// Builds the complete consolidation prompt, ready to send to any LLM in
  /// JSON mode. See [EngramPrompts.buildDreamInstruction] for the template.
  String buildPrompt({EngramLocale locale = EngramLocale.ja}) =>
      EngramPrompts.buildDreamInstruction(
        currentTime: currentLocalTime,
        listing: membersAsJson(),
        locale: locale,
      );
}

/// One proposed replacement memory inside a [DreamDecision].
class DreamProposal {
  /// Creates a proposal.
  const DreamProposal({required this.text, this.timezone});

  /// Self-contained proposition (the engine shortens texts over the soft
  /// limit and discards empty ones).
  final String text;

  /// Optional IANA timezone name suggested by the LLM (informational; the
  /// stored timezone is the one passed to `dream()`).
  final String? timezone;
}

/// The LLM's verdict for one cluster.
class DreamDecision {
  /// Creates a decision.
  const DreamDecision({required this.action, this.memories = const []});

  /// "Leave the cluster untouched."
  const DreamDecision.none()
      : action = DreamAction.none,
        memories = const [];

  /// Parses a raw LLM response leniently: strips code fences, extracts the
  /// first JSON object from surrounding prose, tolerates a missing/unknown
  /// `action` (treated as `merge` if memories are present, else `none`),
  /// and drops entries without text. Never throws — malformed input yields
  /// [DreamDecision.none].
  factory DreamDecision.parseJson(String? raw) {
    final obj = relaxedJsonDecode(raw);
    if (obj is! Map) return const DreamDecision.none();
    final memsRaw = obj['memories'];
    var actionText = (obj['action'] ?? 'none').toString().trim().toLowerCase();
    if (!['merge', 'split', 'none'].contains(actionText)) {
      actionText = (memsRaw is List && memsRaw.isNotEmpty) ? 'merge' : 'none';
    }
    final action = DreamAction.values.byName(actionText);
    if (memsRaw is! List) return DreamDecision(action: action);
    final memories = <DreamProposal>[];
    for (final item in memsRaw) {
      if (item is! Map) continue;
      final text = (item['text'] ?? '').toString().trim();
      if (text.isEmpty) continue;
      final tz = item['timezone']?.toString().trim();
      memories.add(DreamProposal(
          text: text, timezone: (tz == null || tz.isEmpty) ? null : tz));
    }
    return DreamDecision(action: action, memories: memories);
  }

  /// merge / split / none.
  final DreamAction action;

  /// Replacement memories (empty for [DreamAction.none]).
  final List<DreamProposal> memories;
}

/// Best-effort JSON parse tolerant of code fences and surrounding prose.
/// Returns the decoded object, or `null` when nothing parseable is found.
Object? relaxedJsonDecode(String? text) {
  if (text == null || text.trim().isEmpty) return null;
  var t = text.trim();
  t = t
      .replaceAll(RegExp(r'^```(?:json)?', multiLine: true), '')
      .replaceAll(RegExp(r'```$', multiLine: true), '')
      .trim();
  try {
    return jsonDecode(t);
  } on FormatException {
    final m = RegExp(r'\{.*\}|\[.*\]', dotAll: true).firstMatch(t);
    if (m != null) {
      try {
        return jsonDecode(m.group(0)!);
      } on FormatException {
        return null;
      }
    }
  }
  return null;
}
