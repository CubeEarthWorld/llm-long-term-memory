"""core.llm_client — robustness of the JSON parsing helpers.

The dream / extract paths must tolerate code fences, surrounding prose and
malformed output without raising (the LLM output is untrusted).
"""
from __future__ import annotations

from core.llm_client import _loads_relaxed, _parse_dream, _parse_texts


def test_loads_relaxed_plain_json():
    assert _loads_relaxed('{"a": 1}') == {"a": 1}


def test_loads_relaxed_code_fence():
    assert _loads_relaxed('```json\n{"a": 1}\n```') == {"a": 1}


def test_loads_relaxed_surrounding_prose():
    assert _loads_relaxed('Sure! Here you go: {"a": 1} hope it helps') == {"a": 1}


def test_loads_relaxed_garbage_returns_none():
    assert _loads_relaxed("not json at all") is None
    assert _loads_relaxed("") is None
    assert _loads_relaxed(None) is None


def test_parse_texts_from_memories_key():
    assert _parse_texts('{"memories": ["a", "b"]}') == ["a", "b"]


def test_parse_texts_from_bare_list():
    assert _parse_texts('["x", "y"]') == ["x", "y"]


def test_parse_texts_from_dicts():
    assert _parse_texts('{"memories": [{"text": "hi"}, {"text": ""}, {"text": "yo"}]}') == ["hi", "yo"]


def test_parse_texts_garbage():
    assert _parse_texts("nope") == []
    assert _parse_texts(None) == []


def test_parse_dream_merge():
    out = _parse_dream('{"action": "merge", "memories": [{"text": "gist", "timezone": "Asia/Tokyo"}]}')
    assert out["action"] == "merge"
    assert out["memories"][0]["text"] == "gist"


def test_parse_dream_none_normalises_empty_memories():
    out = _parse_dream('{"action": "none", "memories": []}')
    assert out["action"] == "none"
    assert out["memories"] == []


def test_parse_dream_unknown_action_with_memories_becomes_merge():
    out = _parse_dream('{"action": "frobnicate", "memories": [{"text": "x"}]}')
    assert out["action"] == "merge"


def test_parse_dream_unknown_action_without_memories_becomes_none():
    out = _parse_dream('{"action": "frobnicate"}')
    assert out["action"] == "none"


def test_parse_dream_drops_empty_text_entries():
    out = _parse_dream('{"action": "merge", "memories": [{"text": "  "}, {"text": "keep"}]}')
    assert [m["text"] for m in out["memories"]] == ["keep"]


def test_parse_dream_garbage_safe():
    assert _parse_dream("garbage")["action"] == "none"
    assert _parse_dream(None)["action"] == "none"
