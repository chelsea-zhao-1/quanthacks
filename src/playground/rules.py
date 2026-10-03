"""Selection rules: the scan refuses to run until selection_rules.json has no TBD left and is committed to Git
with no uncommitted changes (CLAUDE.md: selection rules are committed before the scan runs)."""
import hashlib
import json
import subprocess
from pathlib import Path

from . import ROOT

RULES_FILE = ROOT / "selection_rules.json"


class RulesNotReady(RuntimeError):
    pass


def _tbd_paths(obj, path="") -> list[str]:
    if isinstance(obj, dict):
        return [p for k, v in obj.items() if not k.startswith("_") for p in _tbd_paths(v, f"{path}.{k}" if path else k)]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in _tbd_paths(v, f"{path}[{i}]")]
    return [path] if isinstance(obj, str) and obj.strip().upper() == "TBD" else []


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def load(path: Path = RULES_FILE, require_committed: bool = True) -> dict:
    """The rules, plus provenance (`_commit`, `_sha256`). Raises RulesNotReady if anything is missing."""
    if not path.exists():
        raise RulesNotReady(f"{path.name} does not exist")
    raw = path.read_bytes()
    rules = json.loads(raw)
    missing = _tbd_paths(rules)
    if missing:
        raise RulesNotReady(f"{path.name} still has TBD for: {', '.join(missing)}")
    rules["_sha256"] = hashlib.sha256(raw).hexdigest()
    rules["_commit"] = None
    if require_committed:
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if _git("ls-files", "--error-unmatch", rel).returncode != 0:
            raise RulesNotReady(f"{rel} is not committed to Git")
        if _git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
            raise RulesNotReady(f"{rel} has uncommitted changes; commit them before the scan")
        rules["_commit"] = _git("log", "-1", "--format=%H", "--", rel).stdout.strip()
    return rules


def git_head() -> str:
    return _git("rev-parse", "HEAD").stdout.strip()
