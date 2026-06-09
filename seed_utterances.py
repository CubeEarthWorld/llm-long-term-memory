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


_SEED_ENTRIES: list[dict[str, str]] = [
    # 1  real "now" at seed start
    {"text": "こんにちは。今日は何をしようかな。", "note": "ノイズ（挨拶＋独り言）", "advance": "0"},
    # 2  same moment
    {"text": "私はアイスクリームが好きです。特に抹茶味が好物です。", "note": "嗜好（やや重要・抹茶）", "advance": "0"},
    # 3  introduce the forgettable fact F1
    {"text": "学生時代はずっとバスケットボール部に所属していました。", "note": "経歴（後で思い出すF1・中重要）", "advance": "30m"},
    # 4
    {"text": "今日はいい天気ですね。", "note": "ノイズ（天気）", "advance": "2h"},
    # 5  meeting (Thu)
    {"text": "来週の木曜日に重大な新商品の開発会議が入っています。", "note": "重要（会議・来週木）", "advance": "6h"},
    # 6  codename
    {"text": "新商品のコードネームは「あおぞら」です。", "note": "重要（固有名・後で長期保持/思い出し）", "advance": "1d"},
    # 7
    {"text": "明日は晴れみたいです。", "note": "ノイズ（天気予報）", "advance": "1d"},
    # 8  meeting moved (next-week Wed) — newer should win
    {"text": "新商品の開発会議は再来週の水曜日に移動しました。", "note": "重要更新（会議・再来週水）", "advance": "2d"},
    # 9
    {"text": "うーん、何をしようかな。", "note": "ノイズ（独り言）", "advance": "1d"},
    # 10
    {"text": "マインクラフトで松明ってどうやって作るんだっけ。", "note": "質問（保存優先度低）", "advance": "0"},
    # 11 confirmation question (still within ~a week of the meeting)
    {"text": "再来週に会議って入っていましたっけ？", "note": "確認質問（更新後の会議日を想起できるか）", "advance": "0"},
    # 12 long absence begins
    {"text": "おひさしぶり、最近どうしてる？しばらく話してなかったね。", "note": "ノイズ（長期経過の起点・+5年）", "advance": "5y"},
    # 13 ~10y total elapsed -> F1 / codename decay into the archive
    {"text": "今日もいい天気だなあ。特に予定はないかな。", "note": "ノイズ（さらに+5年＝累計約10年経過）", "advance": "5y"},
    # 14 cue F1 back from the archive (思い出し)
    {"text": "ところで、私が学生時代にやっていた部活って何だっけ？", "note": "思い出しテスト（部活F1をアーカイブから復元できるか）", "advance": "0"},
    # 15 cue codename back from the archive (思い出し)
    {"text": "新商品「あおぞら」のことって、まだ覚えてる？", "note": "思い出しテスト（固有名の長期保持/復元）", "advance": "0"},
    # 16 restate preference -> reinforce / savings
    {"text": "私はアイスクリームが好きです。", "note": "重複再出現（near-dup補強 or savings復元）", "advance": "30d"},
    # 17 ask the preference back
    {"text": "私の好きな食べ物は何か覚えてる？", "note": "嗜好の想起テスト", "advance": "0"},
]

SEED_UTTERANCES: list[str] = [e["text"] for e in _SEED_ENTRIES]
SEED_NOTES: list[str] = [e["note"] for e in _SEED_ENTRIES]
SEED_ADVANCE: list[str] = [e["advance"] for e in _SEED_ENTRIES]
