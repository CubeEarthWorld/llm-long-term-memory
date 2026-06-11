import datetime
import sys

print("=== ordinal constant ===")
print("date(1970,1,1).toordinal() =", datetime.date(1970, 1, 1).toordinal())
print("code uses 719528 for that")

from memory.long_term_memory import _ymd_from_ordinal, _fmt_local

print("\n=== _ymd_from_ordinal vs datetime.date.fromordinal ===")
bad = 0
for ordn in [1, 366, 719163, 719528, 730000, 1000000, 2000000, 3652059]:
    got = _ymd_from_ordinal(ordn)
    exp = datetime.date.fromordinal(ordn)
    expt = (exp.year, exp.month, exp.day)
    flag = "OK" if got == expt else "MISMATCH"
    if got != expt:
        bad += 1
    print(f"ord={ordn:>8}  got={got}  expected={expt}  {flag}")
print("mismatches:", bad)

print("\n=== _fmt_local for a normal time ===")
# 2026-06-11 ~ unix 1781000000
print(_fmt_local(1749600000, "Asia/Tokyo;+09:00"))

print("\n=== _fmt_local for year > 9999 (manual path) ===")
# year 9999 ~ 2.534e11; pick something larger to force manual path
big = 300000000000  # ~ year 11000
print("unix", big, "->", _fmt_local(big, "UTC;+00:00"))
# Compare to what the day count implies
days = big // 86400
print("days since epoch:", days)

print("\n=== _filter_by_score behavior ===")
from config import default_config
cfg = default_config()
print("score_thresholds =", cfg.memory.score_thresholds)

print("\n=== count(turn_log) ===")
from core.storage import Store
import tempfile, os
tmp = tempfile.mkdtemp()
st = Store(os.path.join(tmp, "t.db"))
st.execscript("CREATE TABLE IF NOT EXISTS turn_log(turn INTEGER PRIMARY KEY, utterance TEXT NOT NULL, note TEXT, timestamp REAL, system_json TEXT);")
try:
    print("count(turn_log) =", st.count("turn_log"))
except Exception as e:
    print("count(turn_log) RAISED:", type(e).__name__, e)
