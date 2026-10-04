"""Regression check of the pipeline's insample path against the committed 2024-25 run (data/oldnews/).

Runs src/oldnews/pipeline.py's run_oldnews on 2024-01-01..2025-12-31, label insample, from the API cache only, with
every output in a temporary folder (deleted afterwards; the real ledger and data/oldnews/ are untouched), and checks
that it reproduces data/oldnews/classified_insample.csv (the old/surprise label of every filing: 0 differences) and
results_insample/h1.csv (the H1 effect and the number of old-news events). It prints only pass or fail, counts and
those two headline values, which are already in the committed results.

Needs the API cache and the committed tables; skips (exit 0, with a message) if they are not there.
Run:  .venv/Scripts/python.exe tests/test_oldnews_pipeline.py
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA = ROOT / "data" / "oldnews"
REF_CLASSIFIED = DATA / "classified_insample.csv"
REF_H1 = DATA / "results_insample" / "h1.csv"


def main() -> int:
    if not (REF_CLASSIFIED.exists() and REF_H1.exists() and (ROOT / ".massive_cache").exists()):
        print("SKIP: the committed insample tables or the API cache are not here")
        return 0
    from oldnews import pipeline as P

    kw = dict(keep_default_na=False, na_values=["", "nan", "NaN"], dtype={"row_id": str})
    ref = pd.read_csv(REF_CLASSIFIED, **kw)
    ref_h1 = pd.read_csv(REF_H1, keep_default_na=False, na_values=["", "nan", "NaN"]).iloc[0]
    ledger_before = (DATA / "ledger.csv").read_bytes() if (DATA / "ledger.csv").exists() else b""

    scratch = Path(tempfile.mkdtemp(prefix="oldnews_regress_"))
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            import nb
            NB = nb.load()
        res = P.run_oldnews("2024-01-01", "2025-12-31", "insample", allow_fetch=False, NB=NB, out_dir=scratch,
                            quiet=True, figures=False)
        got = res["classified"]
        got_h1 = res["tests"]["h1"].iloc[0]
        src = got["t_source"].value_counts().to_dict() if "t_source" in got else {}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    wall = time.time() - t0

    fails = []
    m = ref[["row_id", "old"]].merge(got[["row_id", "old"]], on="row_id", how="outer", suffixes=("_ref", "_got"),
                                     indicator=True)
    only = int((m["_merge"] != "both").sum())
    both = m[m["_merge"] == "both"]
    flag = lambda s: s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])  # noqa: E731
    diff = int((flag(both["old_ref"]) != flag(both["old_got"])).sum())
    if only or diff:
        fails.append(f"old labels: {diff} differ, {only} rows in only one table")
    if not np.isclose(float(got_h1["effect"]), float(ref_h1["effect"]), atol=1e-9):
        fails.append(f"H1 effect {float(got_h1['effect']):+.4f} vs committed {float(ref_h1['effect']):+.4f}")
    if int(got_h1["n_old"]) != int(ref_h1["n_old"]) or int(got_h1["n"]) != int(ref_h1["n"]):
        fails.append(f"H1 n {int(got_h1['n'])} (old {int(got_h1['n_old'])}) vs committed {int(ref_h1['n'])} "
                     f"(old {int(ref_h1['n_old'])})")
    if ((DATA / "ledger.csv").read_bytes() if (DATA / "ledger.csv").exists() else b"") != ledger_before:
        fails.append("the real ledger changed")
    if scratch.exists():
        fails.append("scratch folder not deleted")

    print(f"pipeline insample path vs committed run: {len(both)} filings compared, {diff} old labels differ; "
          f"H1 {float(got_h1['effect']):+.4f} with {int(got_h1['n_old'])} old (committed {float(ref_h1['effect']):+.4f}, "
          f"{int(ref_h1['n_old'])} old); word-score source {src}; {wall / 60:.1f} min")
    print("ALL PASS" if not fails else "FAILED: " + "; ".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
