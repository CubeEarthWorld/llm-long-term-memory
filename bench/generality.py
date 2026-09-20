"""汎用性の検証: 系は特定の埋め込みモデル・次元に依存していないか。

EmbeddingGemma は Matryoshka (MRL) なので、先頭 d 次元で切って再正規化すれば
「別次元の別モデル」になる。同じ年表・同じ本文で d を振り、(1) コサイン分布、
(2) 想起品質と淘汰精度、(3) cue による夢のクラスタ捕捉率を測る。閾値
(cosineFloor / thetaRelated / gistMinCosine) だけがモデル依存であることを、
較正ありと較正なしの両方で示す。

  python bench/generality.py --dims 768 512 256 128 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.forget_bench import (BENCH, CACHE, CachedProvider, build_cache, load_corpus,  # noqa: E402
                                load_updates, run)
from bench.supersede import cluster_capture  # noqa: E402
from config import Config  # noqa: E402
from memory.util import cues  # noqa: E402

OUT = os.path.join(BENCH, "generality_{model}.json")


class Truncated:
    """MRL 切り詰め provider。base の出力の先頭 dim 次元だけを使い、再正規化する。
    model_id に次元を含める（次元を変えてモデル名を変えないと、古いベクトルが
    再索引されずに次元不一致のまま残る）。"""

    def __init__(self, base, dim: int):
        self.base, self.dimension = base, dim
        self.model_id = f"{base.model_id}#d{dim}"

    def _cut(self, M: np.ndarray) -> np.ndarray:
        V = M[:, : self.dimension].astype(np.float32)
        n = np.linalg.norm(V, axis=1, keepdims=True)
        return V / np.where(n == 0.0, 1.0, n)

    def encode_document(self, texts):
        return self._cut(self.base.encode_document(texts))

    def encode_query(self, texts):
        return self._cut(self.base.encode_query(texts))

    def pop_ms(self) -> float:
        return 0.0


def _cfg(src: Config) -> Config:
    """年表 1 本ごとに手つかずの設定を作る（run は capacity などを書き換える）。"""
    c = Config()
    c.glob = src.glob
    return c


def distribution(p, corpus: dict, rng: np.random.Generator) -> dict:
    """無関係な文どうしのコサイン（= cosineFloor の較正点）と、probe→gold のコサイン。"""
    pool = [(g["fact"], cues(g["probes"][0], 8)) for g in corpus["gold"]]
    facts = [f for f, c in pool if c]
    probes = [c[0] for _, c in pool if c]      # キャッシュの鍵は cues() で割った手がかり
    D, Q = p.encode_document(facts), p.encode_query(probes)
    same = np.sum(D * Q, axis=1)                                   # 正解ペア
    i = rng.permutation(len(facts))
    i = np.where(i == np.arange(len(facts)), (i + 1) % len(facts), i)
    cross = np.sum(D[i] * Q, axis=1)                               # 無関係ペア
    pct = lambda v, q: [round(float(x), 3) for x in np.percentile(v, q)]  # noqa: E731
    return {"unrelated_p50_p90_p99": pct(cross, [50, 90, 99]),
            "matching_p10_p50_p90": pct(same, [10, 50, 90]),
            "separation": round(float(np.median(same) - np.percentile(cross, 90)), 3)}


def calibrate(cfg: Config, dist: dict) -> Config:
    """SPEC §6 の較正: 無関係コサインの p90 を cosineFloor、p99 を thetaRelated に置く。
    gistMinCosine は EmbeddingGemma 既定 (0.4 / 0.5 / 0.55) と同じ相対位置に保つ。"""
    _, p90, p99 = dist["unrelated_p50_p90_p99"]
    c = _cfg(cfg)
    floor = round(p90, 2)
    theta = max(round(p99, 2), floor + 0.01)
    c.memory.cosine_floor = floor
    c.memory.theta_related = theta
    c.memory.gist_min_cosine = round(floor + (theta - floor) * 2 / 3, 2)
    return c


def _cfg2(src: Config) -> Config:
    """閾値だけを引き継いだ新しい設定（run が書き換えた capacity 等を持ち込まない）。"""
    c = _cfg(src)
    c.memory.cosine_floor = src.memory.cosine_floor
    c.memory.theta_related = src.memory.theta_related
    c.memory.gist_min_cosine = src.memory.gist_min_cosine
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description="embedding-model / dimension generality")
    ap.add_argument("--model", default="", help="別の GGUF（既定は config の EmbeddingGemma）")
    ap.add_argument("--query-prefix", default=None, help="そのモデルの検索プロンプト（例 'query: '）")
    ap.add_argument("--document-prefix", default=None, help="例 'passage: '")
    ap.add_argument("--dims", type=int, nargs="+", default=[])
    ap.add_argument("--span-days", type=float, default=365.0)
    ap.add_argument("--capacity", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    corpus = load_corpus()
    pairs = load_updates()
    cfg0 = Config()
    cache = CACHE
    if a.model:
        cfg0.glob.embedding_model = a.model
        if a.query_prefix is not None:
            cfg0.glob.embed_query_prefix = a.query_prefix
        if a.document_prefix is not None:
            cfg0.glob.embed_document_prefix = a.document_prefix
        cache = os.path.join(BENCH, f"embeddings_{os.path.splitext(os.path.basename(a.model))[0]}.npz")
        if not os.path.exists(cache):
            build_cache(corpus, cfg0, cache)     # そのモデルで年表の本文を一度だけ埋め込む
    base = CachedProvider(cache)                 # 年表用（実ベクトルの事前計算キャッシュ）
    dims = a.dims or [base.dimension]
    from core.embedding import get_provider      # 更新ペアと cue はキャッシュに無いので実モデルで

    live = get_provider(cfg0.glob)
    print(f"model={base.model_id} dim={base.dimension}")
    print("年表（既定閾値 0.4/0.55/0.5 と、そのモデルで較正した閾値）:", flush=True)
    rows = []
    for dim in dims:
        p = Truncated(base, dim) if dim != base.dimension else base
        dist = distribution(p, corpus, np.random.default_rng(7))
        row = {"dimension": dim, "cosine": dist}
        for label, cfg in (("default", _cfg(cfg0)), ("calibrated", calibrate(cfg0, dist))):
            r = run(corpus, "engram", a.seed, a.span_days, a.capacity, 0.5, 1.0, "keep", cfg, provider=p)
            lp = Truncated(live, dim) if dim != live.dimension else live
            row[label] = {"thresholds": [cfg.memory.cosine_floor, cfg.memory.theta_related,
                                         cfg.memory.gist_min_cosine],
                          "rehearsed": r["final"]["rehearsed"],
                          "dormant_survival": r["final"]["dormant"]["survival"],
                          "evict_precision": r["evict_precision"], "invariants": r["invariants"],
                          "forget_api": r["forget_api"],
                          "capture": cluster_capture(pairs, _cfg2(cfg), lp)}
            x = row[label]
            print(f"  d={dim:4d} {label:10s} floor/θ/gist={x['thresholds']}  "
                  f"top1={x['rehearsed']['top1']:.3f} hit5={x['rehearsed']['hit@5']:.3f} "
                  f"surv={x['rehearsed']['survival']:.3f} evict_prec={x['evict_precision']:.3f} "
                  f"capture={x['capture']['rate']:.0%} bad={sum(x['invariants'].values())}", flush=True)
        print(f"        unrelated p50/p90/p99={dist['unrelated_p50_p90_p99']} "
              f"match p10/p50/p90={dist['matching_p10_p50_p90']} sep={dist['separation']:.3f}", flush=True)
        rows.append(row)

    out = OUT.format(model=os.path.splitext(base.model_id)[0])
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model_id": base.model_id, "dimension": base.dimension, "runs": rows},
                  f, ensure_ascii=False, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
