"""Seed utterances for the LLM Long-Term Memory prototype.

The set deliberately mixes stable facts, low-value noise, and an update sequence,
and spans a *virtual* timeline so the full forgetting pipeline can be exercised
from a single seed run. ``SEED_ADVANCE[i]`` is how far the virtual clock advances
*before* each utterance ``i`` (relative to the previous one); the first entry is "0",
so utterance #1 happens at the real wall-clock time captured at seed start, and
later utterances move cumulatively into the future.

Duration syntax (see ``core.engine._parse_duration``): a number with a unit
suffix — ``s`` seconds, ``m`` minutes, ``h`` hours, ``d`` days, ``w`` weeks,
``y`` years — or a bare number meaning days. "0" means no advance.

What the ENGRAM scenario exercises:
* noise rejection (greetings / weather / small talk) — should not be saved by the LLM;
* identity thresholds (#16 restates #2 → exact/near-dup rehearsal; #5 Thu -> #8 next-week Wed
  is a same-proposition update → tombstone old, insert new, §4.3);
* activation decay + machine movement: #3 (club) and #6 (codename) lose activation across the
  ~10-year gap (#12, #13) and demote toward L2/L3 (768d → 256/128, §5.4), then #14 / #15 still
  recall them — the body text is lossless in every tier;
* cluster formation (meeting / new-product utterances) as material for dream() consolidation (§6).
"""
from __future__ import annotations


_SEED_ENTRIES: list[dict[str, str]] = [
    # 1  real "now" at seed start
    {"text": "こんにちは。今日は何をしようかな。", "advance": "0"},
    # 2  same moment
    {"text": "私はアイスクリームが好きです。特に抹茶味が好物です。", "advance": "0"},
    # 3  introduce the forgettable fact F1
    {"text": "学生時代はずっとバスケットボール部に所属していました。", "advance": "30m"},
    # 4
    {"text": "今日はいい天気ですね。", "advance": "2h"},
    # 5  meeting (Thu)
    {"text": "来週の木曜日に重大な新商品の開発会議が入っています。", "advance": "6h"},
    # 6  codename
    {"text": "新商品のコードネームは「あおぞら」です。", "advance": "1d"},
    # 7
    {"text": "明日は晴れみたいです。", "advance": "1d"},
    # 8  meeting moved (next-week Wed) — newer should win
    {"text": "新商品の開発会議は再来週の水曜日に移動しました。", "advance": "2d"},
    # 9
    {"text": "うーん、何をしようかな。", "advance": "1d"},
    # 10
    {"text": "マインクラフトで松明ってどうやって作るんだっけ。", "advance": "0"},
    # 11 confirmation question (still within ~a week of the meeting)
    {"text": "再来週に会議って入っていましたっけ？", "advance": "0"},
    # 12 long absence begins
    {"text": "おひさしぶり、最近どうしてる？しばらく話してなかったね。", "advance": "5y"},
    # 13 ~10y total elapsed -> F1 / codename decay into the archive
    {"text": "今日もいい天気だなあ。特に予定はないかな。", "advance": "5y"},
    # 14 cue F1 back from the archive (思い出し)
    {"text": "ところで、私が学生時代にやっていた部活って何だっけ？", "advance": "0"},
    # 15 cue codename back from the archive (思い出し)
    {"text": "新商品「あおぞら」のことって、まだ覚えてる？", "advance": "0"},
    # 16 restate preference -> reinforce / savings
    {"text": "私はアイスクリームが好きです。", "advance": "30d"},
    # 17 ask the preference back
    {"text": "私の好きな食べ物は何か覚えてる？", "advance": "0"},
]

SEED_UTTERANCES: list[str] = [e["text"] for e in _SEED_ENTRIES]
SEED_ADVANCE: list[str] = [e["advance"] for e in _SEED_ENTRIES]
