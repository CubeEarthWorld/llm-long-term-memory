"""Seed utterances for the LLM Long-Term Memory prototype.

The set deliberately mixes stable facts, low-value noise, and an update sequence,
and spans a *virtual* timeline so the full forgetting pipeline can be exercised
from a single seed run. ``SEED_ADVANCE[i]`` is how far the virtual clock advances
*before* utterance ``i`` (relative to the previous one); the first entry is "0",
so utterance #1 happens at the real wall-clock time captured at seed start, and
later utterances move cumulatively into the future.

Duration syntax (see ``core.engine._parse_duration``): a number with a unit
suffix — ``s`` seconds, ``m`` minutes, ``h`` hours, ``d`` days, ``w`` weeks,
``y`` years — or a bare number meaning days. "0" means no advance.

What the scenario tests:
* noise rejection (greetings / weather / small talk) — should not be stored, or stored weakly;
* preference & importance weighting (#2 ice cream, #6 codename, #5/#8 meeting);
* update-overrides-old (#5 Thu -> #8 next-week Wed -> #11 confirmation question);
* decay -> archive -> 思い出し (recall): #3 (club) and #6 (codename) fall out of active
  store across the ~10-year gap (#12, #13), then #14 / #15 cue them back from the archive;
* reappearance savings / near-duplicate reinforcement (#16 restates #2);
* cluster formation (meeting / new-product utterances) as material for dream() consolidation.
"""
from __future__ import annotations

from typing import List

SEED_UTTERANCES: List[str] = [
    "こんにちは。今日は何をしようかな。",
    "私はアイスクリームが好きです。特に抹茶味が好物です。",
    "学生時代はずっとバスケットボール部に所属していました。",
    "今日はいい天気ですね。",
    "来週の木曜日に重大な新商品の開発会議が入っています。",
    "新商品のコードネームは「あおぞら」です。",
    "明日は晴れみたいです。",
    "新商品の開発会議は再来週の水曜日に移動しました。",
    "うーん、何をしようかな。",
    "マインクラフトで松明ってどうやって作るんだっけ。",
    "再来週に会議って入っていましたっけ？",
    "おひさしぶり、最近どうしてる？しばらく話してなかったね。",
    "今日もいい天気だなあ。特に予定はないかな。",
    "ところで、私が学生時代にやっていた部活って何だっけ？",
    "新商品「あおぞら」のことって、まだ覚えてる？",
    "私はアイスクリームが好きです。",
    "私の好きな食べ物は何か覚えてる？",
]

SEED_NOTES: List[str] = [
    "ノイズ（挨拶＋独り言）",
    "嗜好（やや重要・抹茶）",
    "経歴（後で思い出すF1・中重要）",
    "ノイズ（天気）",
    "重要（会議・来週木）",
    "重要（固有名・後で長期保持/思い出し）",
    "ノイズ（天気予報）",
    "重要更新（会議・再来週水）",
    "ノイズ（独り言）",
    "質問（保存優先度低）",
    "確認質問（更新後の会議日を想起できるか）",
    "ノイズ（長期経過の起点・+5年）",
    "ノイズ（さらに+5年＝累計約10年経過）",
    "思い出しテスト（部活F1をアーカイブから復元できるか）",
    "思い出しテスト（固有名の長期保持/復元）",
    "重複再出現（near-dup補強 or savings復元）",
    "嗜好の想起テスト",
]

# Per-utterance virtual-clock advance (applied *before* each utterance). First = "0".
SEED_ADVANCE: List[str] = [
    "0",    # 1  real "now" at seed start
    "0",    # 2  same moment
    "30m",  # 3  introduce the forgettable fact F1
    "2h",   # 4
    "6h",   # 5  meeting (Thu)
    "1d",   # 6  codename
    "1d",   # 7
    "2d",   # 8  meeting moved (next-week Wed) — newer should win
    "1d",   # 9
    "0",    # 10
    "0",    # 11 confirmation question (still within ~a week of the meeting)
    "5y",   # 12 long absence begins
    "5y",   # 13 ~10y total elapsed -> F1 / codename decay into the archive
    "0",    # 14 cue F1 back from the archive (思い出し)
    "0",    # 15 cue codename back from the archive (思い出し)
    "30d",  # 16 restate preference -> reinforce / savings
    "0",    # 17 ask the preference back
]
