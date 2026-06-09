"""Unified LLM client for the LLM Long-Term Memory prototype."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

SYSTEM_PROMPT = (
    "あなたは長期記憶を持つ日本語アシスタントです。"
    "想起された記憶だけを過去文脈として扱い、現在のユーザー発話に簡潔に答えてください。"
    "記憶が空、または無関係な場合は無理に参照しないでください。"
)


@dataclass
class LLMResult:
    text: str
    latency_ms: float
    ok: bool
    error: Optional[str] = None
    prompt: str = ""


class LLMClient:
    def __init__(
        self,
        provider: str = "deepseek",
        deepseek_model: str = "deepseek-v4-flash",
        deepseek_base_url: str = "https://api.deepseek.com",
        gemini_model: str = "gemini-3.5-flash",
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
    ):
        self.provider = provider.lower()
        self.deepseek_model = deepseek_model
        self.deepseek_base_url = deepseek_base_url
        self.gemini_model = gemini_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.init_error: Optional[str] = None
        self._client = None
        self._init()

    @property
    def model(self) -> str:
        return self.deepseek_model if self.provider == "deepseek" else self.gemini_model

    def _init(self) -> None:
        try:
            if self.provider == "deepseek":
                key = os.getenv("DEEPSEEK_API_KEY")
                if not key:
                    raise RuntimeError("DEEPSEEK_API_KEY is not set. Check .env.")
                from openai import OpenAI

                self._client = OpenAI(api_key=key, base_url=self.deepseek_base_url)
            elif self.provider == "gemini":
                key = os.getenv("GEMINI_API_KEY")
                if not key:
                    raise RuntimeError("GEMINI_API_KEY is not set. Check .env.")
                from google import genai

                self._client = genai.Client(api_key=key)
            else:
                raise RuntimeError(f"Unknown provider: {self.provider}")
        except Exception as e:  # noqa: BLE001
            self.init_error = f"{type(e).__name__}: {e}"
            self._client = None

    @property
    def status(self) -> str:
        if self._client is None:
            return f"ERROR - {self.init_error}"
        return f"OK - {self.provider}:{self.model}"

    def _build_prompt(self, memory_pack: str, user_text: str) -> str:
        pack = memory_pack.strip() or "(関連する記憶なし)"
        return (
            f"# 想起された記憶\n{pack}\n\n"
            f"# ユーザーの発話\n{user_text}\n\n"
            "# あなたの応答（簡潔に）"
        )

    def respond(self, memory_pack: str, user_text: str) -> LLMResult:
        """Generate an assistant response given recalled memories and the current user utterance."""
        prompt = self._build_prompt(memory_pack, user_text)
        if self._client is None:
            return LLMResult("", 0.0, False, self.init_error, prompt)
        t0 = time.perf_counter()
        try:
            if self.provider == "deepseek":
                text = self._deepseek_chat(SYSTEM_PROMPT, prompt, json_mode=False)
            else:
                text = self._gemini_chat(SYSTEM_PROMPT, prompt)
            dt = (time.perf_counter() - t0) * 1000.0
            return LLMResult(text.strip(), dt, True, None, prompt)
        except Exception as e:  # noqa: BLE001
            dt = (time.perf_counter() - t0) * 1000.0
            return LLMResult(f"[LLM error] {type(e).__name__}: {e}", dt, False, str(e), prompt)

    def extract_memory(self, user_text: str, assistant_text: str) -> List[Dict]:
        """Ask the LLM to extract storable facts from a user/assistant exchange. Returns a list of memory dicts."""
        if self._client is None:
            return []
        instruction = (
            "次のユーザー発話とアシスタント応答から、長期記憶に保存すべき重要情報だけを抽出してください。\n"
            "保存対象は、ユーザーの好み、名前、所属、継続的な予定や制約、明示的な指示、"
            "あとで役立つ安定した事実です。\n"
            "保存しない対象は、挨拶、一時的な雑談、天気のような一般知識、単発の質問です。\n"
            "1つの発話に独立した複数の事実が含まれる場合は、無理に1件へまとめず、"
            "事実ごとに別々の memory に分割してください（1 memory = 1 つの事実）。"
            "保存すべき事実が1つだけのとき、または無いときは、1件のみ、または空配列にしてください。\n"
            "JSON オブジェクト {\"memories\": [...]} のみを返してください。該当なしは {\"memories\": []}。\n"
            f"各 memory の形式: {_EXTRACT_SCHEMA_LONG_MEMORY}\n\n"
            f"# ユーザー発話\n{user_text}\n\n"
            f"# アシスタント応答\n{assistant_text}\n"
        )
        try:
            if self.provider == "deepseek":
                raw = self._deepseek_chat(
                    "You are a memory extraction engine. Return JSON only.",
                    instruction,
                    json_mode=True,
                    temperature=0.0,
                )
            else:
                raw = self._gemini_chat(
                    "You are a memory extraction engine. Return JSON only.",
                    instruction,
                    json_mode=True,
                    temperature=0.0,
                )
            return _parse_memories(raw)
        except Exception:
            return []

    def dream_cluster(self, members: List[Dict]) -> Dict:
        """Sleep-like consolidation of one cluster's memories.

        ``members`` carry text, importance ``w``, provenance, the content time
        (``updated_at_local`` + ``updated_at_unix`` + ``timezone``) and activation
        ``freq``. The LLM chooses to merge, split, or do nothing, and writes the
        resulting memories (which replace the inputs). Returns
        ``{"action": "merge|split|none", "memories": [...]}``.
        """
        if self._client is None or not members:
            return {"action": "none", "memories": []}
        listing = json.dumps(members, ensure_ascii=False, indent=2)
        instruction = (
            "あなたは長期記憶を「夢を見る」ように整理する統合エンジンです。\n"
            "以下は同じ意味クラスタに属する記憶の一覧です。各記憶には内容時刻"
            "(updated_at_local / updated_at_unix / timezone)と重要度 w、保持率 r(0-1, 想起しやすさ) があります。\n\n"
            "次のいずれかを選んでください:\n"
            "- merge: 重複や関連する記憶を、より少数の記憶へ統合・抽象化する。"
            "細かい具体やエピソードの枝葉は削り、後で役立つ要点(gist)を残す。"
            "古くなった内容は現在時刻を踏まえて更新する"
            "(例『7月に旅行予定』→『2026年7月に旅行済み』)。矛盾は新しい時刻のものを優先。\n"
            "- split: 1つの記憶に複数の事実が詰め込まれている場合、独立した記憶へ分割する。\n"
            "- none: 整理が不要なら何もしない。\n\n"
            "無理に1つへまとめる必要はありません。異なる事実は別々の記憶として残してください"
            "(例: 10件を3件にする等)。\n"
            "新しい各記憶には text, w(0.0-1.0), provenance(user|inferred), "
            "timezone(IANA名 例 Asia/Tokyo) を必ず書いてください。\n"
            "出力は JSON オブジェクトのみ: "
            '{"action": "merge|split|none", "memories": [{"text": "...", "w": 0.7, '
            '"provenance": "user", "timezone": "Asia/Tokyo"}]}\n'
            "action が none のときは memories を空配列にしてください。\n\n"
            f"# クラスタ内の記憶\n{listing}\n"
        )
        try:
            sys_prompt = "You are a memory consolidation engine. Return JSON only."
            if self.provider == "deepseek":
                raw = self._deepseek_chat(sys_prompt, instruction, json_mode=True, temperature=0.2)
            else:
                raw = self._gemini_chat(sys_prompt, instruction, json_mode=True, temperature=0.2)
            return _parse_dream(raw)
        except Exception:
            return {"action": "none", "memories": []}

    def _deepseek_chat(self, system: str, user: str, json_mode: bool, temperature=None) -> str:
        """Provider-specific chat completion for DeepSeek (OpenAI-compatible endpoint)."""
        kwargs = dict(
            model=self.deepseek_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_output_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def _gemini_chat(self, system: str, user: str, json_mode: bool = False, temperature=None) -> str:
        """Provider-specific chat completion for Google Gemini."""
        from google.genai import types

        cfg = dict(
            system_instruction=system,
            temperature=self.temperature if temperature is None else temperature,
            max_output_tokens=self.max_output_tokens,
        )
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        resp = self._client.models.generate_content(
            model=self.gemini_model,
            contents=user,
            config=types.GenerateContentConfig(**cfg),
        )
        return resp.text or ""


_EXTRACT_SCHEMA_LONG_MEMORY = (
    '{"text": "保存する短い記憶", "w": 0.0-1.0 の重要度, '
    '"provenance": "user または inferred"}'
)


def _parse_dream(text: Optional[str]) -> Dict:
    """Parse a dreaming decision: {"action": ..., "memories": [...]}."""
    obj = _loads_relaxed(text)
    if not isinstance(obj, dict):
        return {"action": "none", "memories": []}
    action = str(obj.get("action", "none")).strip().lower()
    if action not in ("merge", "split", "none"):
        action = "merge" if obj.get("memories") else "none"
    mems = obj.get("memories")
    mems = [m for m in mems if isinstance(m, dict) and m.get("text")] if isinstance(mems, list) else []
    return {"action": action, "memories": mems}


def _loads_relaxed(text: Optional[str]):
    """Best-effort JSON parse tolerant of code fences and surrounding prose."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}|\[.*\]", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def _parse_memories(text: Optional[str]) -> List[Dict]:
    """Best-effort parse of memory-extraction JSON into a list of validated memory dicts."""
    if not text:
        return []
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    data = None
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}|\[.*\]", text, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                return []
    if data is None:
        return []
    if isinstance(data, dict):
        if "memories" in data and isinstance(data["memories"], list):
            data = data["memories"]
        else:
            data = [data]
    return [d for d in data if isinstance(d, dict) and d.get("text")]
