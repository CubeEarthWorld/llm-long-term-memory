"""Vector & activation operations for :class:`LongTermMemory` (§3.3, §4.1, §8).

Mixin: methods run on the engine instance (``self.store`` / ``self.memory`` /
``self.provider`` are provided by ``LongTermMemory.__init__``).
"""
from __future__ import annotations

import numpy as np

from core.embedding import l2_normalize, truncate_normalize
from core.storage import dequantize_int8, pack_vec, quantize_int8, unpack_vec
from memory.helpers import ulid

# Max entries in the per-id 768d re-embed cache used by _identity_scan (§5.1).
_DOC_VEC_CACHE_MAX = 4096


def decode_unit_vec(blob: bytes, dtype: str, scale: float | None) -> np.ndarray:
    """Decode a stored vector blob (f32 or int8+scale) into a unit-norm float32 array."""
    v = unpack_vec(blob) if dtype == "f32" else dequantize_int8(blob, scale)
    return l2_normalize(np.asarray(v, dtype=np.float32))


class VectorOpsMixin:
    # ================================================================== #
    # activation A (§4.1)
    # ================================================================== #
    def _tau(self, tier: int) -> float:
        return (self.memory.tau1, self.memory.tau2, self.memory.tau3)[int(tier) - 1]

    def _dim(self, tier: int) -> int:
        return (self.memory.dim1, self.memory.dim2, self.memory.dim3)[int(tier) - 1]

    def activation(self, row, now: float) -> float:
        """A(now) = mass · 2^(−max(0, now−last_access)/τ_tier). Underflow → 0 (§4.1, §8.1)."""
        mass = float(row["mass"])
        dt = max(0.0, now - float(row["last_access"]))
        exp = dt / self._tau(int(row["tier"]))
        if exp > self.memory.decay_exp_cap:
            return 0.0
        return min(mass * (2.0 ** (-exp)), self.memory.m_max)

    def _apply_recall(self, row, now: float) -> None:
        """Recall update with the §4.1 ordering: fold decay first, then refractory bonus."""
        mass = self.activation(row, now)              # 1) fold decay into mass
        lb = float(row["last_bonus_at"])
        if now - lb >= self.memory.refractory_seconds:  # 2) refractory gate
            mass = min(mass + 1.0, self.memory.m_max)
            lb = now
        self.store.exec(
            "UPDATE memory SET mass=?, last_access=?, last_bonus_at=? WHERE id=?",
            (mass, int(now), int(lb), row["id"]),
        )

    def _reinforce_write(self, row, now: float) -> None:
        """Exact-text rehearsal (§5.1.2): definite +1, bypassing the refractory gate."""
        mass = min(self.activation(row, now) + 1.0, self.memory.m_max)
        self.store.exec(
            "UPDATE memory SET mass=?, last_access=?, last_bonus_at=? WHERE id=?",
            (mass, int(now), int(now), row["id"]),
        )

    # ================================================================== #
    # vectors (§3.3, §8 MRL+int8)
    # ================================================================== #
    def _embed_doc(self, text: str) -> np.ndarray:
        return np.asarray(self.provider.encode_document(text, dim=self.glob.dim_full)[0], dtype=np.float32)

    def _store_vec(self, mid: str, v_full: np.ndarray, tier: int) -> None:
        dim = self._dim(tier)
        vt = truncate_normalize(np.asarray(v_full, dtype=np.float32), dim)
        if tier == 1:
            blob, dtype, scale = pack_vec(vt), "f32", None
        else:
            blob, scale = quantize_int8(vt)
            dtype = "int8"
        self.store.exec(
            "INSERT OR REPLACE INTO vec(id,memory_id,model_id,dim,dtype,scale,v) VALUES(?,?,?,?,?,?,?)",
            (ulid(), mid, self.model_id, dim, dtype, scale, blob),
        )

    @staticmethod
    def _vec_array(vrow) -> np.ndarray:
        """Decode a stored vec row into a unit-norm float32 vector."""
        return decode_unit_vec(vrow["v"], vrow["dtype"], vrow["scale"])

    def _tier_candidates(self, tier: int) -> tuple[list, np.ndarray | None]:
        """Return (rows, unit-norm vector matrix) for live memories in ``tier`` (joins vec).

        The matrix form lets callers score a whole tier with one ``M @ q``
        instead of a per-row Python loop of ``np.dot`` calls.
        """
        rows = self.store.query(
            "SELECT m.*, v.dim AS _dim, v.dtype AS _dtype, v.scale AS _scale, v.v AS _v "
            "FROM memory m JOIN vec v ON v.memory_id=m.id AND v.model_id=? "
            "WHERE m.tier=? AND m.superseded_by IS NULL",
            (self.model_id, tier),
        )
        if not rows:
            return [], None
        vecs = [decode_unit_vec(r["_v"], r["_dtype"], r["_scale"]) for r in rows]
        return list(rows), np.stack(vecs)

    @staticmethod
    def _coarse128(vec: np.ndarray) -> np.ndarray:
        return truncate_normalize(np.asarray(vec, dtype=np.float32), 128)

    def _doc_vec_cached(self, row) -> np.ndarray:
        """768d document vector for precision refinement, cached per memory id.

        Safe because text is immutable for a given id (edits create a new row);
        without this, every write re-embeds the same L2/L3 neighbours. Cleared
        wholesale at the size cap so the cache itself stays bounded.
        """
        mid = row["id"]
        v = self._doc_vec_cache.get(mid)
        if v is None:
            if len(self._doc_vec_cache) >= _DOC_VEC_CACHE_MAX:
                self._doc_vec_cache.clear()
            v = self._embed_doc(row["text"])
            self._doc_vec_cache[mid] = v
        return v

