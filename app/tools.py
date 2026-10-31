"""Tools exposed to the agent, defined with LangChain's ``@tool`` decorator.

These illustrate the "agent frameworks" requirement: the agent can decide,
based on the user's request, whether it needs to call a deterministic tool
(e.g. a calculator) instead of hallucinating an answer -- a core idea behind
LangChain/LlamaIndex/AutoGen-style agents.
"""
from __future__ import annotations

import ast
import operator as op
import shlex
import subprocess

from langchain_core.tools import tool

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'.

    Use this whenever the user asks for a numeric computation so the answer
    is exact rather than guessed by the LLM.
    """
    try:
        tree = ast.parse(expression, mode="eval").body
        return str(_safe_eval(tree))
    except Exception:
        return "Error: could not evaluate expression"


@tool
def knowledge_base_lookup(topic: str) -> str:
    """Look up a short fact about a known internal topic.

    This simulates a retrieval tool (as you'd build with LlamaIndex over a
    real document/vector store) without requiring external infrastructure.
    """
    mock_kb = {
        "refund policy": "Refunds are processed within 5-7 business days of approval.",
        "sla": "Standard support SLA is a 4 business hour first response time.",
        "pricing": "Pricing tiers are Starter, Growth, and Enterprise, billed monthly.",
    }
    return mock_kb.get(topic.strip().lower(), f"No internal knowledge found for '{topic}'.")


# Security model for the shell tool: an ALLOWLIST of read-only, informational
# command *names* (not a denylist of "bad" ones -- denylists are notoriously
# easy to bypass). Combined with shlex.split() + subprocess with shell=False
# (so there's no shell metacharacter injection, e.g. "; rm -rf /" is just
# treated as a literal, harmless argument) and a hard timeout. This is the
# single biggest real-world risk area in "agents with OS access": an LLM can
# be tricked via prompt injection into running something destructive, so the
# tool itself -- not the LLM's good intentions -- has to enforce the boundary.
ALLOWED_SHELL_COMMANDS = {
    "pwd",
    "date",
    "whoami",
    "uname",
    "hostname",
    "uptime",
    "df",
    "ls",
    "echo",
}

SHELL_COMMAND_TIMEOUT_SECONDS = 5
SHELL_OUTPUT_MAX_CHARS = 2000


@tool
def run_shell_command(command: str) -> str:
    """Run a safe, read-only OS command and return its output.

    Only informational commands are allowed (e.g. 'pwd', 'date', 'whoami',
    'uname -a', 'df -h', 'ls -la'). Anything else (file writes, process
    control, network access, package installs, etc.) is rejected before
    anything is executed. Use this for questions like "what's the current
    date/time", "what OS/user is this running as", or "how much disk space
    is left" -- not as a general-purpose shell.
    """
    try:
        args = shlex.split(command)
    except ValueError as exc:
        return f"Error: could not parse command ({exc})"

    if not args:
        return "Error: empty command"

    program = args[0]
    if program not in ALLOWED_SHELL_COMMANDS:
        allowed = ", ".join(sorted(ALLOWED_SHELL_COMMANDS))
        return f"Error: '{program}' is not an allowed command. Allowed commands: {allowed}"

    try:
        result = subprocess.run(
            args,
            shell=False,  # critical: args are passed directly to exec(), no shell parsing/injection
            capture_output=True,
            text=True,
            timeout=SHELL_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {SHELL_COMMAND_TIMEOUT_SECONDS}s"
    except FileNotFoundError:
        return f"Error: '{program}' is not available on this system"

    output = (result.stdout or "") + (result.stderr or "")
    output = output.strip() or "(command produced no output)"
    if len(output) > SHELL_OUTPUT_MAX_CHARS:
        output = output[:SHELL_OUTPUT_MAX_CHARS] + "... [truncated]"
    return output


AVAILABLE_TOOLS = [calculator, knowledge_base_lookup, run_shell_command]
