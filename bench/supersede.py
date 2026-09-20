"""上書きによる忘却（SPEC §5 dream）のベンチマーク。

事実が更新されたとき、古い版が本当に消えるか。old を書き、30日後に new を書くと
ペアは labile になり、dream が keep / replace を裁定する。裁定だけは実 LLM が要るので
ここは本番と同じ ``LLMClient``（DeepSeek）を呼ぶ。

  python bench/supersede.py --analyze          # LLM なし: old/new の cos とクラスタ捕捉率の診断
  python bench/supersede.py --run              # 実 LLM で夢を見せて stale 率を測る
  python bench/supersede.py --run --theta 0.65 # θ_related を変えた場合

計測は `frozen` の中で行うので、probe 自体が記憶を強化することはない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.forget_bench import BENCH, _DAY, frozen, load_updates  # noqa: E402
from config import Config  # noqa: E402
from core.storage import Store  # noqa: E402
from memory.engine import LongTermMemory  # noqa: E402

OUT = os.path.join(BENCH, "supersede_result")
_GAP_DAYS = 30.0


def scenario(pairs: list[dict], llm, cfg: Config, provider=None):
    """old を全部書き、30日進めて new を全部書く（SPEC の labile ペア）。dream が書く gist も
    埋め込む必要があるので、ここは実 EmbeddingGemma を直に使う（~360文、数秒）。"""
    from core.embedding import get_provider

    t = [1_700_000_000.0]
    s = LongTermMemory(Store(":memory:"), provider or get_provider(cfg.glob), llm,
                       cfg.memory, "Asia/Tokyo", lambda: t[0])
    for p in pairs:
        t[0] += 600
        s.remember(p["old"], cue=p["cue_old"])
    t[0] += _GAP_DAYS * _DAY
    for p in pairs:
        t[0] += 600
        s.remember(p["new"], cue=p["cue_new"])
    return s, t


def cluster_capture(pairs: list[dict], cfg: Config, provider=None) -> dict:
    """cue で引き直した夢のクラスタが old と new を同席させた割合（generality.py と共用）。"""
    s, _ = scenario(pairs, None, cfg, provider=provider)
    clusters = [{m.text for m in c} for c in s.clusters()]
    got = sum(1 for x in pairs if any({x["old"], x["new"]} <= c for c in clusters))
    s.store.close()
    return {"captured": got, "pairs": len(pairs), "rate": round(got / len(pairs), 3),
            "clusters": len(clusters)}


def analyze(pairs: list[dict], cfg: Config) -> None:
    """上書き忘却の律速は LLM ではなくクラスタ捕捉: old と new の cos が θ_related を超えるか。"""
    import numpy as np

    from core.embedding import get_provider

    p = get_provider(cfg.glob)
    cos = [float(a @ b) for a, b in zip(p.encode_document([x["old"] for x in pairs]),
                                        p.encode_document([x["new"] for x in pairs]))]
    q = np.percentile(cos, [10, 25, 50, 75, 90])
    print("cos(old,new): min %.3f p10 %.3f p25 %.3f median %.3f p75 %.3f p90 %.3f max %.3f"
          % (min(cos), *q, max(cos)))
    for th in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        hit = sum(c >= th for c in cos)
        print(f"  theta_related={th:.2f} → cos で到達 {hit:3d}/{len(cos)} ({hit / len(cos):.0%})")
    print("")
    print("実際に dream が同じクラスタへ入れたペア数（cue で引き直すので cos(old,new) とは別物）:")
    for th in (0.45, 0.50, 0.55, 0.60, 0.65):
        for members in (8, 16):
            cfg.memory.theta_related, cfg.memory.dream_max_members = th, members
            c = cluster_capture(pairs, cfg)
            print(f"  theta={th:.2f} max_members={members:2d} → {c['captured']:3d}/{c['pairs']} "
                  f"({c['rate']:.0%})  clusters={c['clusters']}")


def measure(pairs: list[dict], cfg: Config) -> dict:
    from core.llm_client import LLMClient

    llm = LLMClient(cfg.glob)
    if llm.init_error:
        raise SystemExit(f"LLM unavailable: {llm.init_error}")
    print(f"live dream via {cfg.glob.llm_provider}/{cfg.glob.deepseek_model}", flush=True)
    s, t = scenario(pairs, llm, cfg)
    before = {m.text for m in s.memories()}
    reports = s.dream(budget=10 ** 6)
    retry = s.dream(budget=10 ** 6)              # 失敗した種は次の夢で再試行される（SPEC §5）
    after = {m.text for m in s.memories()}
    gists = after - before

    def count(rs, action):
        return sum(1 for r in rs if r["action"] == action)

    stats = {"pairs": len(pairs), "theta_related": cfg.memory.theta_related, "clusters": len(reports),
             "replace": count(reports, "replace"), "keep": count(reports, "keep"),
             "error": count(reports, "error"), "gists": len(gists),
             "retry_clusters": len(retry), "retry_replace": count(retry, "replace"),
             "retry_keep": count(retry, "keep"), "retry_error": count(retry, "error"),
             "errors": sorted({r["error"] for r in reports + retry if r["action"] == "error"})[:5]}

    judged = [{r["text"] for r in rep["before"]} for rep in reports + retry]   # 夢が実際に裁定した集合
    groups: dict[bool, list[dict]] = {True: [], False: []}
    samples = []
    with frozen(s):
        for p in pairs:
            pack = [r["text"] for r in s.recall(p["probe"], now=t[0])["recalled"]]
            captured = any({p["old"], p["new"]} <= c for c in judged)     # old と new が同じ夢に入ったか
            groups[captured].append({"old_alive": p["old"] in after, "stale": p["old"] in pack,
                                     "current": p["new"] in pack or any(g in pack for g in gists),
                                     "empty": not pack})
            if len(samples) < 12:
                samples.append({"probe": p["probe"], "captured": captured,
                                "old_in_pack": p["old"] in pack, "top": pack[:2]})

    def rate(rows: list[dict], field: str):
        return round(sum(r[field] for r in rows) / len(rows), 4) if rows else None

    stats.update({"captured": len(groups[True]), "not_captured": len(groups[False])})
    for label, rows in (("", groups[True] + groups[False]), ("captured_", groups[True]), ("missed_", groups[False])):
        stats.update({label + "old_survival": rate(rows, "old_alive"), label + "stale_in_pack": rate(rows, "stale"),
                      label + "current_in_pack": rate(rows, "current"), label + "empty_pack": rate(rows, "empty")})
    stats["samples"] = samples
    s.store.close()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="overwrite-forgetting (dream) benchmark")
    ap.add_argument("--analyze", action="store_true", help="LLM なしで cos 分布とクラスタ捕捉率を出す")
    ap.add_argument("--run", action="store_true", help="実 LLM に夢を見させて stale 率を測る")
    ap.add_argument("--theta", type=float, default=0.0, help="theta_related を上書きする")
    a = ap.parse_args()
    cfg = Config()
    if a.theta:
        cfg.memory.theta_related = a.theta
    pairs = load_updates()
    if a.analyze:
        analyze(pairs, cfg)
        return
    if not a.run:
        raise SystemExit("--analyze か --run を指定してください")
    res = measure(pairs, cfg)
    with open(f"{OUT}_theta{cfg.memory.theta_related:g}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
