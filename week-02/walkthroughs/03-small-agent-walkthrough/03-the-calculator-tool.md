# Step 2 — The Calculator Tool

> [Back to index](README.md) · Previous: [Environment Setup](02-environment-setup.md) · Next: [Web Search, Save Note, and the Registry](04-web-search-save-note-and-the-registry.md)

## Goal

Build `tools.py` with its first tool: `calculator`, implemented as a restricted AST walk instead of a call to Python's `eval()`.

## Why this matters

The obvious way to build an arithmetic tool for a model is `eval(expression)`. It's also a real vulnerability the moment the "expression" comes from model-generated text: a confused or adversarially-prompted model could produce something like `"__import__('os').system('rm -rf ~')"`, and a bare `eval()` would run it as regular Python — no different from executing arbitrary code an attacker handed you.

The fix isn't "sanitize the string" — string sanitization against arbitrary Python is famously easy to get wrong. Instead, this tool parses the expression into an AST (`ast.parse`) and walks it with a function that only knows how to handle a small, explicit allowlist of node types: numeric constants, the six binary arithmetic operators, and unary plus/minus. Every other node type — function calls, attribute access, names, imports, anything — falls through to a `raise ValueError`. The tool can *only ever* compute arithmetic, no matter what string it's handed, because nothing else in the AST is even recognized, let alone executed.

Create the file:

```bash
touch tools.py
```

## 1. Module docstring and imports

```python
"""Tool definitions for the small agent.

Three tools, deliberately chosen to cover both halves of the idempotent
vs. side-effecting distinction:

- `calculator` and `web_search` are read-only / idempotent — safe to
  auto-retry on a transient failure.
- `save_note` writes a file — a side-effecting action that must NOT be
  blindly retried (a retried "save" after an ambiguous failure could write
  a duplicate note).

Each tool is a contract: a name, a typed Pydantic schema, and a
predictable, bounded return shape — not just a bare Python function.
"""
import ast
import operator

from pydantic import BaseModel, Field
```

The docstring describes the whole module even though only one of its three tools exists yet — write it now, since the design decision (idempotent vs. side-effecting) is what shapes every tool you're about to add, not something that emerges after the fact.

## 2. The argument schema

```python
# --------------------------------------------------------------------------
# calculator — idempotent
# --------------------------------------------------------------------------


class CalculatorArgs(BaseModel):
    expression: str = Field(..., description="A numeric arithmetic expression, e.g. '17 * 12.99'")
```

Same pattern as Step 2's `WebSearchArgs`: one required field, with a `description` that later gets rendered straight into the system prompt.

## 3. The allowlist

```python
# Only these AST node types are allowed — this is what makes `calculator`
# safe to expose to a model at all. A bare `eval()` on model-generated text
# would let it run arbitrary Python (e.g. "__import__('os').system(...)");
# walking a restricted AST instead means the tool can only ever compute
# arithmetic, no matter what string the model sends.
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
```

Two dictionaries, each mapping an AST operator node type to the actual Python function that implements it. Nothing here executes code from the input yet — this is just the lookup table `_eval_node` will consult below.

## 4. The restricted walk

```python
def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")
```

Walk through the four recognized cases with trainees, then the fallthrough:

1. `ast.Expression` — the wrapper `ast.parse(..., mode="eval")` always produces; unwrap it and recurse into its body.
2. `ast.Constant` holding an `int`/`float` — a literal number; the recursion's base case.
3. `ast.BinOp` whose operator is in the allowlist — recursively evaluate both sides, then apply the matching function.
4. `ast.UnaryOp` whose operator is in the allowlist — recursively evaluate the operand, then negate/pose it.
5. Anything else — a function call, a name lookup, a string, an import, a lambda — `raise ValueError`. This is the security boundary: it's a default-deny fallthrough, not a blocklist of specific dangerous constructs, so nothing needs to be anticipated in advance.

## 5. The public function

```python
def calculator(expression: str) -> float:
    """Evaluate a numeric expression using a restricted AST walk (safe subset
    of Python arithmetic — no names, calls, attribute access, or imports)."""
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_node(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as e:
        # A predictable, typed failure — the dispatcher feeds this back to
        # the model as a correctable argument error.
        raise ValueError(f"could not evaluate '{expression}': {e}") from e
```

Every failure mode — an unparseable string (`SyntaxError`), a disallowed construct (`ValueError` from `_eval_node`), dividing by zero (`ZeroDivisionError`), or a type mismatch — is normalized into a single `ValueError` with a message naming the original expression. That uniform failure type is what a later dispatcher step will catch and turn into a correctable observation.

## Try it

```bash
uv run python -c "
from tools import calculator
print(calculator('17 * 12.99'))
try:
    calculator('__import__(\"os\").system(\"echo pwned\")')
except ValueError as e:
    print('blocked:', e)
"
```

Expected output:

```text
220.83
blocked: could not evaluate '__import__("os").system("echo pwned")': Unsupported expression element: Call(func=Name(id='__import__', ctx=Load()), args=[Constant(value='os')], keywords=[])
```

The second call never runs `os.system` — it fails at the AST-walk stage, before any code from the string is ever executed.

## Checkpoint

<details>
<summary>Full <code>tools.py</code> (calculator only)</summary>

```python
"""Tool definitions for the small agent.

Three tools, deliberately chosen to cover both halves of the idempotent
vs. side-effecting distinction:

- `calculator` and `web_search` are read-only / idempotent — safe to
  auto-retry on a transient failure.
- `save_note` writes a file — a side-effecting action that must NOT be
  blindly retried (a retried "save" after an ambiguous failure could write
  a duplicate note).

Each tool is a contract: a name, a typed Pydantic schema, and a
predictable, bounded return shape — not just a bare Python function.
"""
import ast
import operator

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# calculator — idempotent
# --------------------------------------------------------------------------


class CalculatorArgs(BaseModel):
    expression: str = Field(..., description="A numeric arithmetic expression, e.g. '17 * 12.99'")


# Only these AST node types are allowed — this is what makes `calculator`
# safe to expose to a model at all. A bare `eval()` on model-generated text
# would let it run arbitrary Python (e.g. "__import__('os').system(...)");
# walking a restricted AST instead means the tool can only ever compute
# arithmetic, no matter what string the model sends.
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def calculator(expression: str) -> float:
    """Evaluate a numeric expression using a restricted AST walk (safe subset
    of Python arithmetic — no names, calls, attribute access, or imports)."""
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_node(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as e:
        # A predictable, typed failure — the dispatcher feeds this back to
        # the model as a correctable argument error.
        raise ValueError(f"could not evaluate '{expression}': {e}") from e
```

</details>

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `calculator('2 ** 10')` raises `ValueError: Unsupported expression element` | `ast.Pow` accidentally left out of `_ALLOWED_BINOPS` | Confirm all six operators from the checkpoint are present |
| A call like `calculator('abs(-5)')` is expected to work but doesn't | Function calls are deliberately never in the allowlist — that's the entire point | Not a bug; the tool only does arithmetic, on purpose |
| `TypeError` escapes instead of being wrapped in `ValueError` | Forgot `TypeError` in the `except` tuple in `calculator()` | Match the exception tuple in the checkpoint exactly |

Next: **[Web Search, Save Note, and the Registry](04-web-search-save-note-and-the-registry.md)** — add the other two tools and the classification that decides how their failures get handled.
