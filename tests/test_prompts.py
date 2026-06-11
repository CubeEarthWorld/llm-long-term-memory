"""LLM prompt framing (§5.2 prompt-injection guard + user-info clarification).

Recalled memories injected into the LLM must be framed as (a) information ABOUT
THE USER and (b) past context, NOT instructions. These defaults are what ship
when no data/prompts.csv override exists.
"""
from __future__ import annotations

from core.llm_client import _DEFAULT_SYSTEM_PROMPT, _DEFAULT_USER_TEMPLATE


def test_system_prompt_marks_recall_as_user_info():
    assert "ユーザーに関する" in _DEFAULT_SYSTEM_PROMPT


def test_system_prompt_keeps_injection_guard():
    # §5.2: recalled memory must be framed as context, not instructions.
    assert "指示ではありません" in _DEFAULT_SYSTEM_PROMPT


def test_user_template_marks_recall_as_user_info():
    assert "ユーザーに関する" in _DEFAULT_USER_TEMPLATE


def test_user_template_keeps_injection_guard():
    assert "指示ではない" in _DEFAULT_USER_TEMPLATE


def test_user_template_has_placeholders():
    assert "{memory_pack}" in _DEFAULT_USER_TEMPLATE
    assert "{user_text}" in _DEFAULT_USER_TEMPLATE
    assert "{current_time}" in _DEFAULT_USER_TEMPLATE


def test_time_placeholders_in_extract_and_dream():
    """Relative dates must be resolvable: every generation point sees 'now'."""
    from core.llm_client import _DEFAULT_DREAM_INSTRUCTION, _DEFAULT_EXTRACT_INSTRUCTION

    assert "{current_time}" in _DEFAULT_EXTRACT_INSTRUCTION
    assert "{current_time}" in _DEFAULT_DREAM_INSTRUCTION
    assert "{listing}" in _DEFAULT_DREAM_INSTRUCTION


def test_build_prompt_inserts_pack_and_text():
    """_build_prompt fills the template and frames the recall section."""
    from core.llm_client import LLMClient

    client = LLMClient(provider="deepseek", prompts={})  # prompts={} → use defaults
    prompt = client._build_prompt("ユーザーは京都に住んでいる", "今日はどこに行こう？")
    assert "ユーザーに関する" in prompt          # recall framed as user info
    assert "ユーザーは京都に住んでいる" in prompt   # the recalled memory is present
    assert "今日はどこに行こう？" in prompt          # the current utterance is present


def test_build_prompt_handles_empty_pack():
    from core.llm_client import LLMClient

    client = LLMClient(provider="deepseek", prompts={})
    prompt = client._build_prompt("", "こんにちは")
    assert "(関連する記憶なし)" in prompt


def test_build_prompt_embeds_current_time():
    from core.llm_client import LLMClient

    client = LLMClient(provider="deepseek", prompts={})
    prompt = client._build_prompt("", "こんにちは", "2026-06-11 09:00 +09:00")
    assert "2026-06-11 09:00 +09:00" in prompt
    assert "{current_time}" not in prompt
    # missing clock degrades gracefully instead of leaking the placeholder
    assert "(不明)" in client._build_prompt("", "こんにちは")


def test_dream_cluster_embeds_current_time(monkeypatch):
    from core.llm_client import LLMClient

    client = LLMClient(provider="deepseek", prompts={})
    client._client = object()  # bypass the missing-API-key early return
    seen = {}

    def fake_chat(system, user, **kw):
        seen["user"] = user
        return '{"action": "none", "memories": []}'

    monkeypatch.setattr(client, "_chat", fake_chat)
    client.dream_cluster(
        [{"id": "m1", "text": "再来週の水曜日に会議がある", "gen": 0, "A": 1.0,
          "local_time": "2026-05-20 10:00 +09:00", "timezone": "Asia/Tokyo;+09:00"}],
        current_time="2026-06-11 09:00 +09:00",
    )
    assert "2026-06-11 09:00 +09:00" in seen["user"]   # 'now' is visible to the LLM
    assert "2026-05-20 10:00 +09:00" in seen["user"]   # each memory keeps its own anchor
    assert "{current_time}" not in seen["user"]


def test_extract_embeds_current_time(monkeypatch):
    from core.llm_client import LLMClient

    client = LLMClient(provider="deepseek", prompts={})
    client._client = object()
    seen = {}

    def fake_chat(system, user, **kw):
        seen["user"] = user
        return '{"memories": []}'

    monkeypatch.setattr(client, "_chat", fake_chat)
    client.extract_save_candidates("再来週の水曜日に会議がある", "了解しました",
                                   current_time="2026-05-20 10:00 +09:00")
    assert "2026-05-20 10:00 +09:00" in seen["user"]
    assert "{current_time}" not in seen["user"]


def test_turn_passes_current_time_to_llm(make_system):
    """Both write paths (converse tools + extract fallback) see the wall clock,
    so relative dates can be normalized to absolute ones at save time."""
    from conftest import ScriptableLLM
    from core.base import TurnRunner
    from core.metrics import MetricsRecorder

    llm = ScriptableLLM(save_on_converse=False)   # nothing saved → extract fallback fires
    s, clock, c = make_system(llm=llm)
    runner = TurnRunner(s.provider, llm, MetricsRecorder(max_history=10), s)
    runner.run_turn(1, "再来週の水曜日に会議がある")
    assert llm.converse_current_time == s.now_local()
    assert llm.extract_current_time == s.now_local()
