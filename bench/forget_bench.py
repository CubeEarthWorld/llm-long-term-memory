"""忘却ベンチマーク（SPEC §3/§4 の減衰・容量エビクションを実測する）。

問い: 「使われた事実は残り、使われない雑談は消える」が本当に起きているか。
容量を超えるまで雑談を書き込み続ける仮想年表を走らせ、節目ごとに凍結して
gold 事実を probe（言い換え質問）で想起できるかを測る。

エビクション方式を差し替えた対照群（fifo / lru / random / none）と同じ年表を
走らせ、ENGRAM の強度順エビクションが baseline より良いことを比較で示す。

使い方:
  python bench/forget_bench.py --build-cache          # 実埋め込みを1回だけ計算
  python bench/forget_bench.py --arm engram --span-days 365 --seed 1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from contextlib import contextmanager

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Every bench script imports this module, so the console encoding is fixed once here.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import Config  # noqa: E402
from core.storage import Store  # noqa: E402
from memory.engine import LongTermMemory  # noqa: E402
from memory.util import cues  # noqa: E402

BENCH = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BENCH, "corpus")
CACHE = os.path.join(BENCH, "embeddings.npz")
_DAY = 86400.0


# ---------------------------------------------------------------- corpus --
def load_corpus(path: str = CORPUS_DIR) -> dict:
    """シャード JSON をまとめ、重複テキストを落とす。"""
    gold, noise, seen = [], [], set()
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            shard = json.load(fh)
        for g in shard.get("gold") or []:
            fact, probes = str(g.get("fact", "")).strip(), [str(p).strip() for p in g.get("probes") or []]
            distractor = str(g.get("distractor", "")).strip()
            if fact and len(probes) >= 1 and fact not in seen:
                seen.add(fact)
                gold.append({"fact": fact, "probes": probes[:2], "distractor": distractor})
        for n in shard.get("noise") or []:
            n = str(n).strip()
            if n and n not in seen:
                seen.add(n)
                noise.append(n)
    for g in gold:                                   # distractor が他の gold と衝突したら捨てる
        if g["distractor"] in seen - {g["distractor"]} and any(o["fact"] == g["distractor"] for o in gold):
            g["distractor"] = ""
    return {"gold": gold, "noise": noise}


def load_updates(path: str = "") -> list[dict]:
    """上書き忘却ベンチの更新ペア（old → new + probe）。``cue_old``/``cue_new`` は各事実だけを
    見て別個に生成させた「後で問い直す質問文」で、本番の save_memory が書くものと同じ立場。"""
    pairs = []
    for f in sorted(glob.glob(os.path.join(path or os.path.join(BENCH, "updates"), "*.json"))):
        with open(f, encoding="utf-8") as fh:
            for p in json.load(fh).get("pairs") or []:
                old, new, probe = (str(p.get(k, "")).strip() for k in ("old", "new", "probe"))
                if old and new and probe and old != new:
                    pairs.append({"old": old, "new": new, "probe": probe,
                                  "cue_old": str(p.get("cue_old", "")).strip(),
                                  "cue_new": str(p.get("cue_new", "")).strip()})
    return pairs


def corpus_texts(corpus: dict, max_cues: int) -> tuple[list[str], list[str]]:
    docs = [g["fact"] for g in corpus["gold"]] + [g["distractor"] for g in corpus["gold"] if g["distractor"]]
    docs += corpus["noise"]
    queries: list[str] = []
    for g in corpus["gold"]:
        for p in g["probes"]:
            queries += cues(p, max_cues)
    return sorted(set(docs)), sorted(set(queries))


# ------------------------------------------------------------ embeddings --
class CachedProvider:
    """事前計算した実 EmbeddingGemma ベクトルを配るだけの provider（モデル不要・即時）。"""

    def __init__(self, path: str = CACHE):
        z = np.load(path)
        self.model_id = str(z["model_id"])
        self._doc = {k: v for k, v in zip(z["doc_keys"].tolist(), z["doc_vecs"])}
        self._q = {k: v for k, v in zip(z["q_keys"].tolist(), z["q_vecs"])}
        self.dimension = int(z["doc_vecs"].shape[1])
        self.total_ms = 0.0

    def _get(self, table: dict, texts) -> np.ndarray:
        missing = [t for t in texts if t not in table]
        if missing:
            raise KeyError(f"embedding cache miss: {missing[0]!r} — rebuild with --build-cache")
        return np.stack([table[t] for t in texts]).astype(np.float32)

    def encode_document(self, texts) -> np.ndarray:
        return self._get(self._doc, texts)

    def encode_query(self, texts) -> np.ndarray:
        return self._get(self._q, texts)

    def pop_ms(self) -> float:
        return 0.0


def build_cache(corpus: dict, cfg: Config, path: str = CACHE) -> None:
    from core.embedding import get_provider

    p = get_provider(cfg.glob)
    docs, queries = corpus_texts(corpus, cfg.memory.max_cues)
    print(f"embedding {len(docs)} docs + {len(queries)} queries with {p.model_id} …", flush=True)
    np.savez(path, model_id=np.array(p.model_id), doc_keys=np.array(docs), q_keys=np.array(queries),
             doc_vecs=p.encode_document(docs), q_vecs=p.encode_query(queries))
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB, {p.pop_ms() / 1000:.0f}s)")


# ------------------------------------------------------------------ arms --
def install_arm(s: LongTermMemory, arm: str, rng: random.Random) -> None:
    """ENGRAM 以外は `_enforce_capacity` を差し替えた対照群。"""
    if arm in ("engram", "none"):
        return
    pick = {"fifo": lambda rows: min(rows, key=lambda m: m.created_at),
            "lru": lambda rows: min(rows, key=lambda m: m.last_recall),
            "random": lambda rows: rows[rng.randrange(len(rows))]}[arm]

    def evict(now: float):
        victim = None
        while len(s._traces) > s.cfg.capacity:
            victim = pick(list(s._traces.values()))
            s._remove(victim.id)
        return victim

    s._enforce_capacity = evict


@contextmanager
def frozen(s: LongTermMemory):
    """計測用: recall の副作用（強化・commit）を止めて状態を凍結する。"""
    put, remove = s._put, s._remove
    s._put = lambda m: None
    s._remove = lambda mid: None
    try:
        yield
    finally:
        s._put, s._remove = put, remove


# --------------------------------------------------------------- measure --
def measure(s: LongTermMemory, corpus: dict, ids: dict, rehearsed: set, now: float) -> dict:
    """凍結した状態で全 gold を probe し、生存率と想起順位を集計する。"""
    alive = {m.text for m in s.memories()}
    out = {"records": len(alive), "day": round((now - ids["t0"]) / _DAY)}
    for group, items in (("rehearsed", [g for g in corpus["gold"] if g["fact"] in rehearsed]),
                         ("dormant", [g for g in corpus["gold"] if g["fact"] not in rehearsed])):
        top1 = hit5 = mrr = probes = 0.0
        beaten = 0
        with frozen(s):
            for g in items:
                for p in g["probes"]:
                    texts = [r["text"] for r in s.recall(p, now=now)["recalled"]]
                    probes += 1
                    if g["fact"] in texts:
                        rank = texts.index(g["fact"]) + 1
                        top1 += rank == 1
                        hit5 += 1
                        mrr += 1.0 / rank
                        if g["distractor"] and g["distractor"] in texts[: rank - 1]:
                            beaten += 1
                    elif g["distractor"] and g["distractor"] in texts:
                        beaten += 1
        n = max(len(items), 1)
        out[group] = {
            "n": len(items),
            "survival": round(sum(g["fact"] in alive for g in items) / n, 4),
            "top1": round(top1 / max(probes, 1), 4),
            "hit@5": round(hit5 / max(probes, 1), 4),
            "mrr": round(mrr / max(probes, 1), 4),
            "distractor_ahead": round(beaten / max(probes, 1), 4),
        }
    out["noise_survival"] = round(sum(n in alive for n in corpus["noise"]) / max(len(corpus["noise"]), 1), 4)
    out["distractor_survival"] = round(
        sum(g["distractor"] in alive for g in corpus["gold"] if g["distractor"])
        / max(sum(1 for g in corpus["gold"] if g["distractor"]), 1), 4)
    return out


def invariants(s: LongTermMemory, now: float) -> dict:
    rows = s.memories()
    rs = [s.retrievability(m, now) for m in rows]
    return {"nan": sum(1 for r in rs if r != r), "r_out_of_range": sum(1 for r in rs if not 0.0 <= r <= 1.0),
            "stability_out_of_range": sum(1 for m in rows if not 1.0 <= m.stability <= s.cfg.max_stability),
            "over_capacity": max(0, len(rows) - s.cfg.capacity), "store_mismatch": s.store.count() - len(rows)}


# ------------------------------------------------------------------- run --
REHEARSALS = (0.002, 0.008, 0.03, 0.1, 0.3, 0.7)   # span に対する拡張間隔（使う事実の再利用）


def run(corpus: dict, arm: str, seed: int, span_days: float, capacity: int,
        rehearsed_frac: float, salience: float, dream: str, cfg: Config, provider=None) -> dict:
    rng = random.Random(seed)
    cfg.memory.capacity = capacity
    provider = provider or CachedProvider()
    store = Store(":memory:")
    t = [1_700_000_000.0]
    s = LongTermMemory(store, provider, _KeepLLM(), cfg.memory, "Asia/Tokyo", lambda: t[0])
    install_arm(s, arm, rng)
    if arm == "none":
        s.cfg.capacity = 10 ** 9

    evicted = {"gold": 0, "distractor": 0, "noise": 0}
    kind = {g["fact"]: "gold" for g in corpus["gold"]}
    kind.update({g["distractor"]: "distractor" for g in corpus["gold"] if g["distractor"]})
    real_remove = s._remove

    def counting_remove(mid: str) -> None:
        m = s._traces.get(mid)
        if m is not None:
            evicted[kind.get(m.text, "noise")] += 1
        real_remove(mid)

    s._remove = counting_remove

    gold = list(corpus["gold"])
    rng.shuffle(gold)
    rehearsed = {g["fact"] for g in gold[: int(len(gold) * rehearsed_frac)]}
    t0 = t[0]
    written_at = {}
    for g in gold:                                       # 年表の起点で gold と distractor を書く
        t[0] += 600
        s.remember(g["fact"], salience=salience)
        written_at[g["fact"]] = t[0]
        if g["distractor"]:
            t[0] += 600
            s.remember(g["distractor"], salience=salience)

    noise = list(corpus["noise"])
    rng.shuffle(noise)
    span = span_days * _DAY
    base = t[0]
    step = span / max(len(noise), 1)

    events = [(base + (i + 1) * step, 0, "noise", text) for i, text in enumerate(noise)]
    for g in gold:                                       # 使われる事実は拡張間隔で再想起される
        if g["fact"] in rehearsed:
            events += [(written_at[g["fact"]] + f * span, 1, "rehearse", g) for f in REHEARSALS]
    if dream != "off":
        events += [(base + k * span / 40, 2, "dream", None) for k in range(1, 41)]
    events += [(base + f * span, 3, "measure", None) for f in (0.01, 0.1, 0.3, 0.6, 1.0)]
    events.sort(key=lambda e: (e[0], e[1]))

    series, rehearsals, cited = [], 0, 0
    for when, _, kindof, payload in events:
        t[0] = when
        if kindof == "noise":
            s.remember(payload, salience=1.0)
        elif kindof == "rehearse":
            rehearsals += 1
            res = s.recall(payload["probes"][0])
            mid = next((r["id"] for r in res["recalled"] if r["text"] == payload["fact"]), None)
            if mid:
                cited += 1
                s.cite(f"《id:{mid}》")                    # 実際に使った＝完全な強化
        elif kindof == "dream":
            s.dream(budget=2)
        else:
            series.append(measure(s, corpus, {"t0": t0}, rehearsed, t[0]))

    result = {"arm": arm, "seed": seed, "span_days": span_days, "capacity": capacity,
              "model_id": provider.model_id, "dimension": provider.dimension,
              "salience": salience, "rehearsed_frac": rehearsed_frac, "dream": dream,
              "corpus": {"gold": len(gold), "noise": len(noise),
                         "distractor": sum(1 for g in gold if g["distractor"])},
              "written": len(gold) + len(noise) + sum(1 for g in gold if g["distractor"]),
              "rehearsals": rehearsals, "rehearsal_hits": cited,
              "evicted": dict(evicted), "final": series[-1] if series else {}, "series": series,
              "invariants": invariants(s, t[0])}
    total = sum(evicted.values())
    result["evict_precision"] = round(evicted["noise"] / total, 4) if total else None
    result["forget_api"] = _forget_api_check(s, corpus, rehearsed, t[0])
    s.store.close()
    return result


def _forget_api_check(s: LongTermMemory, corpus: dict, rehearsed: set, now: float) -> dict:
    """明示的な忘却（SPEC §4 forget）: 消した事実は二度と想起されない。"""
    alive = {m.text: m.id for m in s.memories()}
    targets = [g for g in corpus["gold"] if g["fact"] in alive][:10]
    for g in targets:
        s.forget(alive[g["fact"]])
    leaked = 0
    with frozen(s):
        for g in targets:
            for p in g["probes"]:
                leaked += g["fact"] in [r["text"] for r in s.recall(p, now=now)["recalled"]]
    return {"deleted": len(targets), "leaked": leaked,
            "store_mismatch": s.store.count() - len(s.memories())}


class _KeepLLM:
    """夢は統合のみ（置換なしで忘却そのものを測る）。"""

    def dream_cluster(self, members, current_time=""):
        return {"action": "keep", "memories": []}


def main() -> None:
    ap = argparse.ArgumentParser(description="ENGRAM v2 forgetting benchmark")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--corpus", default=CORPUS_DIR)
    ap.add_argument("--arm", default="engram", choices=["engram", "fifo", "lru", "random", "none"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--span-days", type=float, default=365.0)
    ap.add_argument("--capacity", type=int, default=500)
    ap.add_argument("--rehearsed-frac", type=float, default=0.5)
    ap.add_argument("--salience", type=float, default=1.0)
    ap.add_argument("--dream", default="keep", choices=["keep", "off"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cfg = Config()
    corpus = load_corpus(a.corpus)
    if a.build_cache:
        build_cache(corpus, cfg)
        return
    res = run(corpus, a.arm, a.seed, a.span_days, a.capacity, a.rehearsed_frac, a.salience, a.dream, cfg)
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
