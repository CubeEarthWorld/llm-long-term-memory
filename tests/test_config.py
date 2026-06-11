"""config — Config (de)serialisation and JSON value coercion."""
from __future__ import annotations

from config import Config, _coerce, default_config


def test_to_dict_from_dict_roundtrip():
    cfg = default_config()
    rebuilt = Config.from_dict(cfg.to_dict())
    assert rebuilt.to_dict() == cfg.to_dict()


def test_from_dict_overrides_subset():
    cfg = Config.from_dict({"memory": {"cap1": 42, "alpha": 0.5}})
    assert cfg.memory.cap1 == 42
    assert cfg.memory.alpha == 0.5
    # untouched values keep their defaults
    assert cfg.memory.cap2 == default_config().memory.cap2


def test_from_dict_ignores_unknown_keys():
    cfg = Config.from_dict({"memory": {"bogus_key": 1}, "glob": {"nope": 2}})
    assert cfg.to_dict() == default_config().to_dict()


def test_coerce_bool():
    assert _coerce("true", False) is True
    assert _coerce(1, False) is True
    assert _coerce("false", True) is False
    assert _coerce(0, True) is False


def test_coerce_int_from_float_string():
    assert _coerce("42.0", 0) == 42
    assert _coerce(3.9, 0) == 3


def test_coerce_float():
    assert _coerce("0.35", 0.0) == 0.35


def test_coerce_tuple_from_list():
    assert _coerce([0.1, 0.2], (0.0,)) == (0.1, 0.2)


def test_coerce_tuple_from_csv_string():
    assert _coerce("0.1, 0.2, 0.3", (0.0,)) == (0.1, 0.2, 0.3)


def test_coerce_bad_value_falls_back_to_default():
    assert _coerce("not-a-number", 5) == 5
    assert _coerce(None, 1.5) == 1.5


def test_score_thresholds_survive_roundtrip():
    cfg = default_config()
    rebuilt = Config.from_dict(cfg.to_dict())
    assert tuple(rebuilt.memory.score_thresholds) == tuple(cfg.memory.score_thresholds)
