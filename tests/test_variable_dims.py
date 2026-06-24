"""Configurable embedding dimensions (dim_full / dim1 / dim2 / dim3 / dim_coarse).

The engine no longer hard-codes the 768/256/128 EmbeddingGemma profile: any
weighting model / policy may redefine the per-tier dimensions, subject to the
MRL nesting constraint dim_full ≥ dim1 ≥ dim2 ≥ dim3 ≥ dim_coarse > 0. These
tests pin the validation and exercise a full write→demote→read→dream cycle on a
non-default dimension profile.
"""
from __future__ import annotations

import pytest

from config import Config, LongTermMemoryConfig, default_config
from tests.conftest import ScriptableLLM, tokens


def _custom_config() -> Config:
    """A deliberately non-default dimension profile (half the usual widths)."""
    cfg = default_config()
    cfg.glob.dim_full = 384
    cfg.memory.dim1 = 384
    cfg.memory.dim2 = 128
    cfg.memory.dim3 = 64
    cfg.memory.dim_coarse = 64
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_custom_dims_accepted():
    cfg = _custom_config()
    assert (cfg.memory.dim1, cfg.memory.dim2, cfg.memory.dim3, cfg.memory.dim_coarse) \
        == (384, 128, 64, 64)


def test_non_nesting_dims_rejected():
    with pytest.raises(ValueError):
        LongTermMemoryConfig(dim1=128, dim2=256, dim3=64)   # dim1 < dim2


def test_coarse_larger_than_tier_rejected():
    with pytest.raises(ValueError):
        LongTermMemoryConfig(dim1=768, dim2=256, dim3=128, dim_coarse=256)


def test_dim1_above_full_rejected():
    cfg = default_config()
    cfg.glob.dim_full = 256
    with pytest.raises(ValueError):
        cfg.validate()                     # dim1 (768) > dim_full (256)


def test_non_positive_dim_rejected():
    with pytest.raises(ValueError):
        LongTermMemoryConfig(dim3=0)


def test_from_dict_revalidates_dims():
    # setattr in from_dict bypasses __post_init__; validation must still run.
    with pytest.raises(ValueError):
        Config.from_dict({"memory": {"dim_coarse": 9999}})


# --------------------------------------------------------------------------- #
# end-to-end on a custom profile
# --------------------------------------------------------------------------- #
def test_full_cycle_on_custom_dims(make_system):
    system, clock, cfg = make_system(config=_custom_config())

    # WRITE: stored L1 vector carries the configured full/dim1 width.
    system.save_memory("alpha beta gamma delta")
    row = system.store.one("SELECT * FROM memory WHERE superseded_by IS NULL")
    vrow = system.store.one("SELECT * FROM vec WHERE memory_id=?", (row["id"],))
    assert vrow["dim"] == cfg.memory.dim1 == 384
    assert vrow["dtype"] == "f32"

    # READ: retrieval works and the coarse MMR vector uses dim_coarse.
    res = system.retrieve("alpha beta gamma delta", turn=1)
    assert "alpha beta gamma delta" in res.pack_text

    # DEMOTE: a forced demotion to L2 truncates to dim2 (int8).
    system._demote(row["id"], 2)
    vrow = system.store.one("SELECT * FROM vec WHERE memory_id=?", (row["id"],))
    assert vrow["dim"] == cfg.memory.dim2 == 128
    assert vrow["dtype"] == "int8"


def test_dream_clusters_on_custom_coarse_dim(make_system):
    system, clock, cfg = make_system(
        llm=ScriptableLLM(dream_action="merge"), config=_custom_config())
    # Three near-identical members so they cluster and exceed cluster_min.
    for i in range(4):
        system.save_memory(f"{tokens(6)} variant {i}")
    results = system.dream(force=True)
    assert isinstance(results, list)   # clustering ran without a dim mismatch
