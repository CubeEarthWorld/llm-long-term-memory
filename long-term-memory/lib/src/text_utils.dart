import 'dart:convert';

import 'package:crypto/crypto.dart';

/// Splits a turn of text into chunks so a memory relevant to any part of a
/// long utterance can surface during retrieval (ENGRAM §5.2).
///
/// Strategy (ported from the reference implementation): blank-line paragraphs
/// first; if that yields ≤1 block, fall back to lines; if still ≤2 blocks and
/// the text is long, split on Japanese/CJK sentence terminators.
List<String> splitParagraphs(String text) {
  var blocks = text
      .split(RegExp(r'\n\s*\n'))
      .map((b) => b.trim())
      .where((b) => b.isNotEmpty)
      .toList();
  if (blocks.length <= 1) {
    blocks = text
        .split('\n')
        .map((l) => l.trim())
        .where((l) => l.isNotEmpty)
        .toList();
  }
  if (blocks.length <= 2 && text.length > 80) {
    final sents = text
        .split(RegExp(r'(?<=[。！？…\n])'))
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();
    if (sents.length > blocks.length) blocks = sents;
  }
  return blocks;
}

/// Truncates [text] to at most [maxChars], preferring a sentence/clause
/// boundary in the second half of the cut (ENGRAM I8 soft limit).
String shorten(String text, int maxChars) {
  if (text.length <= maxChars) return text;
  final cut = text.substring(0, maxChars);
  for (final sep in ['。', '．', '.', '、', ' ']) {
    final idx = cut.lastIndexOf(sep);
    if (idx > maxChars * 0.5) return cut.substring(0, idx + 1);
  }
  return cut;
}

/// Content-address fingerprint of a cluster: SHA-256 of the sorted member
/// ids joined by `'|'` (ENGRAM §6, I12). If membership changes, the
/// fingerprint changes, which self-invalidates the dream-log cache.
String clusterFingerprint(List<String> ids) {
  final sorted = [...ids]..sort();
  return sha256.convert(utf8.encode(sorted.join('|'))).toString();
}
