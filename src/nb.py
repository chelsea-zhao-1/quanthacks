"""Load the starter notebook's own definitions into a namespace, so scripts reuse exactly the code the
judges run instead of a copy that can drift.

Only definitions run: functions, classes, imports and plain assignments from the "library" cells, plus the
configuration and calendar cells in full. Top-level work (the in-sample run, plots, the taxonomy print)
is skipped, and nothing that prints any part of the API key is executed.
"""
import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "gator-quant-hacks-8k-options-challenge.ipynb"

# cell id -> how to run it. "full": every statement. "lib": definitions only. "client": the API client cell,
# everything except top-level print calls (the notebook prints the key's last four characters there).
CELLS = {
    "15d9b7b9": "client",   # 1 · API client
    "c18eb5e0": "full",     # 2 · configuration
    "f95a44d2": "full",     # 3 · trading calendar
    "4088d957": "lib",      # 4 · events
    "97daa32b": "lib",      # 5 · option chain and bars
    "5b64e0d5": "lib",      # 5 · pricing every event
    "1312109b": "lib",      # 6 · P&L engine (STRATEGIES, strategy_pnl, evaluate)
    "3cfc04f4": "lib",      # 7 · scoreboard helpers
    "ffad8ab2": "lib",      # 11 · trade specification (COST_HAIRCUT)
}


def _has_call(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) for n in ast.walk(node))


def _keep(stmt: ast.stmt, mode: str) -> bool:
    if mode == "full":
        return True
    is_print = (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                and getattr(stmt.value.func, "id", None) == "print")
    if mode == "client":
        return not is_print
    if isinstance(stmt, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
        return True
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [getattr(stmt, "target", None)]
    return (isinstance(stmt, (ast.Assign, ast.AnnAssign)) and not _has_call(stmt)
            and all(isinstance(t, ast.Name) for t in targets))       # plain constants, not e.g. events["x"] = ...


def load(cells: dict[str, str] = CELLS) -> dict:
    """Run the selected notebook cells in order and return their namespace."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    by_id = {c.get("id"): c for c in nb["cells"]}
    ns: dict = {"display": print}
    for cell_id, mode in cells.items():
        tree = ast.parse("".join(by_id[cell_id]["source"]))
        for stmt in (s for s in tree.body if _keep(s, mode)):
            code = compile(ast.Module(body=[stmt], type_ignores=[]), f"<notebook cell {cell_id}>", "exec")
            try:
                exec(code, ns)
            except NameError:
                if mode != "lib" or not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                    raise
                # a top-level line that reads the notebook's run results (e.g. hz = results[...]): not a definition
    return ns
