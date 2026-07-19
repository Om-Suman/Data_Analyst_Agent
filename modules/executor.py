"""
Safe Code Execution Sandbox
Validates, sanitizes, and executes AI-generated Pandas code.
"""
import sys
import io
import re
import ast
import time
import traceback
import builtins
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BLOCKED_IMPORTS = {
    "os", "subprocess", "sys", "shutil", "socket", "urllib",
    "http", "ftplib", "smtplib", "pickle", "importlib",
    "__import__", "eval", "exec", "compile", "open",
    "builtins", "ctypes", "multiprocessing", "threading",
}

# Generated code cannot contain import statements, but Pandas and Plotly may
# lazily import their own internals while methods such as Timestamp.strftime()
# run. Expose a narrowly scoped import hook for those library internals rather
# than removing Python's import mechanism completely.
SAFE_IMPORT_ROOTS = {
    "pandas", "numpy", "plotly", "matplotlib", "seaborn", "sklearn", "scipy",
    "dateutil", "pytz", "tzdata", "datetime", "math", "statistics", "re",
    "collections", "functools", "itertools", "typing", "warnings", "json",
    "calendar", "locale", "time", "decimal", "numbers", "copy", "operator",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Allow imports needed by approved analytics libraries, not user code."""
    root = str(name).split(".", 1)[0]
    if root not in SAFE_IMPORT_ROOTS:
        raise ImportError(f"Import of '{root}' is not permitted in the analysis sandbox.")
    return builtins.__import__(name, globals, locals, fromlist, level)

ALLOWED_GLOBALS = {
    "pd": pd, "pandas": pd,
    "np": np, "numpy": np,
    "px": px, "go": go,
    "plt": plt, "matplotlib": matplotlib,
    "sns": sns, "seaborn": sns,
    "print": print,
    "len": len, "range": range, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool,
    "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
    "sorted": sorted, "reversed": reversed,
    "isinstance": isinstance, "type": type,
    "__builtins__": {"__import__": _safe_import},
}

class ExecutionResult:
    def __init__(self):
        self.stdout = ""
        self.error = None
        self.figures = []        # plotly figures
        self.mpl_figures = []    # matplotlib figures
        self.dataframes = {}     # name -> df created during execution
        self.locals = {}
        self.execution_time = 0.0
        self.success = False

def validate_code(code: str) -> tuple[bool, str]:
    """Validate code safety before execution."""
    # Check for blocked imports
    for blocked in BLOCKED_IMPORTS:
        pattern = rf'\b(import\s+{blocked}|from\s+{blocked})'
        if re.search(pattern, code):
            return False, f"Blocked import detected: '{blocked}'"

    # Check for dangerous builtins
    dangerous = ["__import__", "eval(", "exec(", "compile(", "open(", "globals(", "locals("]
    for d in dangerous:
        if d in code:
            return False, f"Dangerous call detected: '{d}'"

    # AST parse check
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    return True, ""


def clean_code(code: str) -> str:
    """Clean up LLM-generated code."""

    # Remove import statements
    code = re.sub(r"^\s*import\s+.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"^\s*from\s+.*?\s+import\s+.*$", "", code, flags=re.MULTILINE)

    # Remove plt.show()
    code = re.sub(r"plt\.show\(\)\s*", "", code)

    # Remove fig.show()
    code = re.sub(r"fig\.show\(\)\s*", "", code)

    # Remove data loading
    code = re.sub(
        r"df\s*=\s*pd\.read_[a-z_]+\(.*?\)\n?",
        "",
        code
    )

    return code.strip()


def execute_code(code: str, df: pd.DataFrame, timeout: int = 30) -> ExecutionResult:
    """Execute code safely and capture all outputs."""
    result = ExecutionResult()
    code = clean_code(code)

    valid, error_msg = validate_code(code)
    if not valid:
        result.error = f"Validation failed: {error_msg}"
        return result

    # Prepare execution namespace
    exec_globals = dict(ALLOWED_GLOBALS)
    exec_globals["df"] = df.copy()

    # Capture stdout
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    # Clear previous matplotlib figures
    plt.close("all")

    start = time.time()
    try:
        exec(code, exec_globals)
        result.success = True
    except Exception:
        result.error = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        result.execution_time = time.time() - start

    result.stdout = buffer.getvalue()

    # Capture matplotlib figures
    for fig_num in plt.get_fignums():
        result.mpl_figures.append(plt.figure(fig_num))

    # Capture plotly figures from locals
    for var_name, val in exec_globals.items():
        if isinstance(val, go.Figure):
            result.figures.append(val)
        elif isinstance(val, pd.DataFrame) and var_name != "df":
            result.dataframes[var_name] = val

    result.locals = {
        k: v for k, v in exec_globals.items()
        if not k.startswith("_") and k not in ALLOWED_GLOBALS
    }

    return result
