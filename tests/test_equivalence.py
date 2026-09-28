"""Same-language regression: the seeded workload must reproduce the outcome 1.0.1
recorded in ``tests/equivalence/golden.json`` bit for bit. Regenerate (only when the
algorithm is meant to change) with ``EQUIVALENCE_WRITE=1 pytest tests/test_equivalence.py``."""
from __future__ import annotations

import json
import os

from workload import run_workload, workload_config

_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "equivalence", "golden.json")


def test_seeded_workload_reproduces_the_golden_outcome(make_system):
    s, clock = make_system(config=workload_config())
    trace = json.loads(json.dumps(run_workload(s, clock), ensure_ascii=False))
    if os.environ.get("EQUIVALENCE_WRITE") == "1":
        os.makedirs(os.path.dirname(_GOLDEN), exist_ok=True)
        with open(_GOLDEN, "w", encoding="utf-8", newline="\n") as f:
            json.dump(trace, f, ensure_ascii=False, indent=1)
            f.write("\n")
    with open(_GOLDEN, encoding="utf-8") as f:
        assert trace == json.load(f)
