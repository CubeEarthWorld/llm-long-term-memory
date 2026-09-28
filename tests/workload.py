"""The seeded workload behind ``tests/equivalence/golden.json``: remember with cues,
recall, cite, clusters, dream with both verdicts, eviction. Recorded with 1.0.1 at full
precision (``repr`` of every float), so a refactor of the search paths must reproduce
1.0.1 bit for bit. Mirrors the Dart package's ``test/support/workload.dart``."""
from __future__ import annotations

import re

_TOPICS = [
    ["kyoto", "trip", "hotel", "train", "temple", "autumn"],
    ["doctor", "clinic", "allergy", "pollen", "medicine", "spring"],
    ["piano", "lesson", "teacher", "recital", "practice", "chopin"],
    ["salary", "bank", "rent", "budget", "savings", "loan"],
    ["cat", "vet", "food", "litter", "toy", "kitten"],
    ["python", "dart", "engine", "memory", "vector", "test"],
]


def workload_config():
    from config import Config
    cfg = Config()
    m = cfg.memory
    m.cosine_floor = 0.0
    m.capacity, m.grace_period, m.dream_budget, m.relative_score = 48, 86400.0, 3, 0.3
    return cfg


def run_workload(s, clock, after_step=None) -> list:
    seed = 0x5EED

    def nxt(n: int) -> int:
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return seed % n

    d = repr
    trace: list = []
    last: dict | None = None
    for _ in range(220):
        clock.advance(600 + nxt(12 * 3600))
        op = nxt(100)
        topic = _TOPICS[nxt(len(_TOPICS))]

        def words(n: int) -> str:
            return " ".join(topic[nxt(6)] for _ in range(n))

        if op < 55:
            text = f"{words(3)} {_TOPICS[nxt(6)][nxt(6)]} n{nxt(40)}"
            cue = "" if nxt(3) == 0 else words(2)
            r = s.remember(text, cue=cue, salience=float(1 + nxt(4)))
            trace.append({"op": "remember", "action": r["action"], "cosine": r.get("cos"),
                          "evicted": r.get("evicted")})
        elif op < 75:
            last = s.recall(f"{words(2)}、{words(2)}")
            trace.append({"op": "recall",
                          "recalled": [[c["text"], d(s._last_recall[c["id"]]["score"]),
                                        d(s._last_recall[c["id"]]["cos"]), d(s._last_recall[c["id"]]["R"])]
                                       for c in last["recalled"]],
                          "pack": re.sub("《id:[^》]+》", "《id》", last["pack_text"])})
        elif op < 80:
            ids = [c["id"] for c in (last or {}).get("recalled", [])]
            trace.append({"op": "cite", "cited": len(s.cite("".join(f"《id:{i}》" for i in ids)))})
        elif op < 90:
            trace.append({"op": "clusters",
                          "clusters": [[m.text for m in c] for c in s.clusters(budget=1 + nxt(4))]})
        else:
            s.llm.action = "replace" if nxt(2) == 0 else "keep"
            reports = s.dream(budget=1 + nxt(3))
            trace.append({"op": "dream", "reports": [
                {"action": rep["action"], "before": [b["text"] for b in rep["before"]],
                 "after": [[a["text"], d(s.memory(a["id"]).stability), s.memory(a["id"]).last_recall]
                           for a in rep["after"]]}
                for rep in reports]})
        if after_step:
            after_step(s)
    trace.append({"op": "final", "memories": [[m.text, d(m.stability), m.last_recall, m.consolidated]
                                              for m in s.memories()]})
    return trace
