"""The discovery playground (CLAUDE.md, "Playground design"): measurements from the cached 2022-2023 download,
matched-null permutation tests, Benjamini-Hochberg, robustness filters and an automatic test ledger.

Nothing here runs on real data until selection_rules.json is complete and committed (see rules.py).
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

HARD_STOP = "2024-01-01"    # no entry or ordinary day at or after this date (discovery window only)
