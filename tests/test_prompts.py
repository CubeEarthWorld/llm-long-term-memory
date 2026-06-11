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
