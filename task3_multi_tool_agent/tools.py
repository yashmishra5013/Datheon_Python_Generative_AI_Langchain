"""
tools.py
--------
Custom LangChain tools used by the autonomous agent:

1. web_search   -> real-time web search (Tavily if key present, else DuckDuckGo)
2. calculator   -> safe Python math evaluator (AST based, no eval())
3. file_reader  -> reads local files from a sandboxed ./workspace folder
"""

from __future__ import annotations

import ast
import math
import operator
import os
from pathlib import Path

from langchain_core.tools import tool

# --------------------------------------------------------------------------- #
# 1. WEB SEARCH
# --------------------------------------------------------------------------- #
MAX_SEARCH_RESULTS = 5


def _search_tavily(query: str) -> str:
    from tavily import TavilyClient  # imported lazily (optional dependency)

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query=query, max_results=MAX_SEARCH_RESULTS)
    items = response.get("results", [])
    return "\n\n".join(
        f"[{i}] {r['title']}\n{r['content']}\nURL: {r['url']}"
        for i, r in enumerate(items, 1)
    )


def _search_duckduckgo(query: str) -> str:
    try:
        from ddgs import DDGS  # newer package name
    except ImportError:  # pragma: no cover
        from duckduckgo_search import DDGS  # older package name

    with DDGS() as ddgs:
        items = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))
    return "\n\n".join(
        f"[{i}] {r.get('title', '')}\n{r.get('body', '')}\nURL: {r.get('href', '')}"
        for i, r in enumerate(items, 1)
    )


@tool
def web_search(query: str) -> str:
    """Search the internet for current, real-time information such as news,
    prices, recent events, facts or anything you do not know. Input must be a
    short, plain-text search query."""
    query = query.strip().strip('"').strip("'")
    if not query:
        return "Error: empty search query."
    try:
        if os.getenv("TAVILY_API_KEY"):
            result = _search_tavily(query)
        else:
            result = _search_duckduckgo(query)
        return result or "No results found. Try rephrasing the query."
    except Exception as exc:  # network / rate-limit issues
        return f"Search failed: {exc}"


# --------------------------------------------------------------------------- #
# 2. CALCULATOR (safe – no eval)
# --------------------------------------------------------------------------- #
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "radians": math.radians, "degrees": math.degrees,
    "factorial": math.factorial, "ceil": math.ceil, "floor": math.floor,
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_MAX_EXPONENT = 10_000


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("Exponent too large.")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*[_eval_node(a) for a in node.args])
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval_node(e) for e in node.elts]
    raise ValueError("Unsupported or unsafe expression.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression precisely. Supports + - * / // % **,
    parentheses, and functions like sqrt, log, log10, sin, cos, tan, factorial,
    round, min, max, sum, plus constants pi and e. Example: 'sqrt(144) + 2**5'.
    Input must be ONLY the expression (no words)."""
    expr = (
        expression.strip().strip("`").strip('"').strip("'")
        .replace("^", "**").replace("×", "*").replace("÷", "/").replace(",000", "000")
    )
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree)
        if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
            result = int(result)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as exc:
        return f"Error: could not evaluate '{expr}' ({exc})."


# --------------------------------------------------------------------------- #
# 3. LOCAL FILE READER (sandboxed to ./workspace)
# --------------------------------------------------------------------------- #
WORKSPACE = Path(__file__).parent.joinpath("workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)
MAX_CHARS = 6000
ALLOWED_SUFFIXES = {".txt", ".md", ".csv", ".json", ".py", ".log", ".yaml", ".yml"}


@tool
def file_reader(file_name: str) -> str:
    """Read a local text file from the agent's workspace folder and return its
    content. Input is a file name such as 'sales.csv' or 'notes.txt'. Use the
    input 'list' to see all available files. Supports txt, md, csv, json, py,
    log, yaml."""
    name = file_name.strip().strip('"').strip("'")

    if name.lower() in {"", "list", ".", "ls"}:
        files = sorted(p.name for p in WORKSPACE.iterdir() if p.is_file())
        return "Available files: " + (", ".join(files) if files else "(none)")

    target = (WORKSPACE / name).resolve()

    # Security: block path traversal (e.g. ../../secret.txt)
    if WORKSPACE not in target.parents and target != WORKSPACE:
        return "Error: access denied. Only files inside the workspace folder can be read."
    if not target.is_file():
        return f"Error: '{name}' not found. Use input 'list' to see available files."
    if target.suffix.lower() not in ALLOWED_SUFFIXES:
        return f"Error: unsupported file type '{target.suffix}'."

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file: {exc}"

    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + f"\n...[truncated, file has {len(text)} characters]"
    return text


ALL_TOOLS = [web_search, calculator, file_reader]
