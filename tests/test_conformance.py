"""Cross-language conformance: replays the Dart package's scripted scenario and
must reproduce its trace (texts, actions, stabilities, scores at 4 decimals)."""
from __future__ import annotations

import json
import os
import re

import pytest

from config import Config
from fakes import FakeEmbeddingProvider

_DART = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "long-term-memory", "test", "conformance")
_DIR = os.environ.get("CONFORMANCE_DIR", _DART)
_NUM = re.compile(r"^-?\d+\.\d{6}$")


def _snake(k: str) -> str:
    return "".join("_" + c.lower() if c.isupper() else c for c in k)


def _norm(v):
    """Numbers (6-decimal strings) are compared at 4 decimals: float32-vs-double order."""
    if isinstance(v, str) and _NUM.match(v):
        return f"{float(v):.4f}"
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


@pytest.mark.skipif(not os.path.exists(os.path.join(_DIR, "trace.json")), reason="Dart trace not generated")
def test_dart_scenario_reproduced(make_system):
    scenario = json.load(open(os.path.join(_DIR, "scenario.json"), encoding="utf-8"))
    expected = json.load(open(os.path.join(_DIR, "trace.json"), encoding="utf-8"))
    cfg = Config.from_dict({"memory": {_snake(k): v for k, v in scenario["config"].items()}})
    cfg.memory.cosine_floor = 0.0
    s, clock = make_system(config=cfg, provider=FakeEmbeddingProvider(dimension=scenario["dimension"]))
    r6 = lambda v: f"{v:.6f}"  # noqa: E731
    trace = []
    for step in scenario["steps"]:
        op = step["op"]
        row = {"op": op}
        if op == "advance":
            clock.advance(step["seconds"])
        elif op == "remember":
            r = s.remember(step["text"], salience=step.get("salience", 1.0))
            m = s.memory(r["id"]) if r.get("id") else None
            row.update(action={"rate_limited": "rateLimited"}.get(r["action"], r["action"]),
                       text=m.text if m else None, stability=r6(m.stability) if m else None,
                       consolidated=m.consolidated if m else None, evicted=r.get("evicted"))
        elif op == "recall":
            r = s.recall(step["query"])
            row.update(texts=[c["text"] for c in r["recalled"]],
                       scores=[r6(s._last_recall[c["id"]]["score"]) for c in r["recalled"]],
                       pack=re.sub("《id:[^》]+》", "《id》", r["pack_text"]))
        elif op == "cite":
            ids = [f"《id:{m.id}》" for m in s.memories() if m.text in step["texts"]]
            row["cited"] = len(s.cite(" ".join(ids)))
        elif op == "dream":
            reports = s.dream(budget=step.get("budget"))
            row["reports"] = [{"action": rep["action"], "before": [b["text"] for b in rep["before"]],
                               "after": [a["text"] for a in rep["after"]],
                               "stability": [r6(s.memory(a["id"]).stability) for a in rep["after"]]}
                              for rep in reports]
        elif op == "forget":
            m = next((m for m in s.memories() if m.text == step["text"]), None)
            row["forgot"] = bool(m) and s.forget(m.id)["action"] == "deleted"
        row["state"] = [{"text": m.text, "S": r6(m.stability), "R": r6(s.retrievability(m, clock.t)),
                         "c": m.consolidated} for m in s.memories()]
        trace.append(row)
    assert len(trace) == len(expected)
    for i, (a, b) in enumerate(zip(trace, expected)):
        assert _norm(a) == _norm(b), f"step {i} ({a['op']}) differs:\n{json.dumps(a, ensure_ascii=False)}\n{json.dumps(b, ensure_ascii=False)}"
