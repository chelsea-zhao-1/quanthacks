"""The test ledger (CLAUDE.md rule 13): every evaluated test is appended, including failures and explorer
queries, so the number of variants can be disclosed. Append-only; one run_id per scan or explorer call."""
import time
import uuid
from pathlib import Path

import pandas as pd

COLUMNS = ["run_id", "timestamp", "git_head", "rules_commit", "rules_sha256", "kind", "level", "group", "subset",
           "filter", "metric", "strategy", "bucket", "horizon", "otm", "entry", "n_sets", "n_tickers",
           "event_mean", "null_mean", "diff", "p_perm", "p_z", "q_bh", "event_mean_2x", "diff_2x", "trimmed_diff",
           "loto_holds", "loto_weakest", "loqo_holds", "loqo_weakest", "plateau_n", "plateau_same",
           "low_sample", "n_perm", "seed"]


def new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def append(path: Path, rows: pd.DataFrame, run_id: str, rules: dict, head: str) -> None:
    rows = rows.assign(run_id=run_id, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), git_head=head,
                       rules_commit=rules.get("_commit"), rules_sha256=rules.get("_sha256"))
    rows = rows.reindex(columns=COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)


def variant_count(path: Path) -> int:
    return 0 if not path.exists() else len(pd.read_csv(path, usecols=["run_id"]))
