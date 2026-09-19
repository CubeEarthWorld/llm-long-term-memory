"""Seed utterances: the built-in scenario, the ``advance`` duration syntax and the
user-editable CSV (``data/seed.csv``, columns text / note / advance).

The scenario mixes stable facts, low-value noise and an update sequence along a
*virtual* timeline. ``advance`` is how far the virtual clock moves *before* the
utterance (relative to the previous one): a number with a unit suffix — ``s``
seconds, ``m`` minutes, ``h`` hours, ``d`` days, ``w`` weeks, ``y`` years (exactly
365 days; no month unit) — or a bare number meaning days. The first utterance
happens at the real wall-clock time captured at seed start.

What the scenario exercises (SPEC §3–§5):
* noise (greetings / weather / small talk) — should not be saved by the LLM;
* #16 restates #2 → exact-text rehearsal; #5 (Thu) → #8 (moved to Wed) is a labile
  pair that the dream adjudicates;
* #3 (club) and #6 (codename) decay across the ~10-year gap (#12, #13), yet #14 / #15
  still recall them — dormant but relevant traces compete;
* the meeting / new-product utterances form a cluster for dream() consolidation.
"""
from __future__ import annotations

import csv
import io
import os

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000}

DEFAULT_SEED: list[dict[str, str]] = [
    {"text": "こんにちは。今日は何をしようかな。", "advance": "0"},
    {"text": "私はアイスクリームが好きです。特に抹茶味が好物です。", "advance": "0"},
    {"text": "学生時代はずっとバスケットボール部に所属していました。", "advance": "30m"},
    {"text": "今日はいい天気ですね。", "advance": "2h"},
    {"text": "来週の木曜日に重大な新商品の開発会議が入っています。", "advance": "6h"},
    {"text": "新商品のコードネームは「あおぞら」です。", "advance": "1d"},
    {"text": "明日は晴れみたいです。", "advance": "1d"},
    {"text": "新商品の開発会議は再来週の水曜日に移動しました。", "advance": "2d"},
    {"text": "うーん、何をしようかな。", "advance": "1d"},
    {"text": "マインクラフトで松明ってどうやって作るんだっけ。", "advance": "0"},
    {"text": "再来週に会議って入っていましたっけ？", "advance": "0"},
    {"text": "おひさしぶり、最近どうしてる？しばらく話してなかったね。", "advance": "5y"},
    {"text": "今日もいい天気だなあ。特に予定はないかな。", "advance": "5y"},
    {"text": "ところで、私が学生時代にやっていた部活って何だっけ？", "advance": "0"},
    {"text": "新商品「あおぞら」のことって、まだ覚えてる？", "advance": "0"},
    {"text": "私はアイスクリームが好きです。", "advance": "10y"},
    {"text": "私の好きな食べ物は何か覚えてる？", "advance": "500y"},
]


def parse_duration(spec: object) -> float:
    """'5y' '8d' '12h' '30m' → seconds; a bare number means days; '0'/''/garbage → 0."""
    s = str(spec or "0").strip().lower()
    unit = _UNITS.get(s[-1:])
    try:
        return float(s[:-1] if unit else s) * (unit or _UNITS["d"])
    except ValueError:
        return 0.0


def clean(raw) -> list[dict[str, str]]:
    """Normalise raw dicts into seed items (text / note / advance); empty texts are dropped."""
    items = []
    for item in raw or []:
        item = item or {}
        text = str(item.get("text") or "").strip()
        if text:
            items.append({"text": text, "note": str(item.get("note") or "").strip(),
                          "advance": str(item.get("advance") or "0").strip() or "0"})
    return items


def parse_csv(text: str) -> list[dict[str, str]]:
    """Tolerant CSV parser: a header row with a ``text`` column, or header-less text[,advance]."""
    text = text.lstrip("﻿")
    reader = csv.DictReader(io.StringIO(text))
    if any((h or "").strip().lower() == "text" for h in reader.fieldnames or []):
        return clean(reader)
    return clean({"text": row[0], "advance": row[1] if len(row) > 1 else "0"}
                 for row in csv.reader(io.StringIO(text)) if row)


def load(path: str) -> list[dict[str, str]]:
    """The saved CSV if present and parseable, otherwise the built-in scenario."""
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            items = parse_csv(f.read())
    except (OSError, csv.Error, UnicodeDecodeError):
        items = []
    return items or clean(DEFAULT_SEED)


def save(path: str, items: list[dict[str, str]]) -> None:
    """Persist seed utterances (UTF-8 with BOM) so edits survive server restarts."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "note", "advance"])
        writer.writerows([i["text"], i["note"], i["advance"]] for i in items)
