"""Tools exposed to the agent.

Where a maintained LangChain-integrated tool/toolkit already exists for a
capability (Wikipedia lookup, file read/write/list), we use that directly
(``langchain_community.tools``/``agent_toolkits``) instead of hand-rolling
one. Where no such integration exists (arithmetic, an internal mock KB,
shell execution, in-place text edits, system resource stats), we write a
small function and expose it with LangChain's ``@tool`` decorator -- the
same mechanism LangChain's own built-in tools use under the hood, so from
the agent's point of view (and in ``AVAILABLE_TOOLS`` below) every tool here
looks and behaves identically regardless of where it came from.

This illustrates the "agent frameworks" requirement: the agent can decide,
based on the user's request, whether it needs to call a deterministic tool
(e.g. a calculator) instead of hallucinating an answer -- a core idea behind
LangChain/LlamaIndex/AutoGen-style agents.
"""
from __future__ import annotations

import ast
import operator as op
import subprocess
from pathlib import Path

import psutil
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool

from app.codebase_rag import codebase_search
from app.postgres_rag import article_search
from app.rag import rag_search
from app.readme_rag import readme_search

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


# Instead of hand-writing an HTTP client + response parser for an external
# knowledge source, this uses LangChain's PREBUILT ``WikipediaQueryRun`` tool
# (from langchain_community.tools) -- a ready-made, agent-ready ``BaseTool``
# built on top of ``WikipediaAPIWrapper`` -- rather than calling the API
# wrapper by hand. It's still exposed to the agent through our own thin
# @tool wrapper so that network/SSL/API errors degrade gracefully instead of
# crashing the agent loop (``WikipediaQueryRun`` itself only swallows
# ``ToolException``s, not arbitrary network errors).
_wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
)


@tool
def wikipedia_search(query: str) -> str:
    """Look up a GENERAL-knowledge topic (people, places, companies, concepts,
    historical events) on Wikipedia and return a short summary.

    Use this for external, real-world knowledge questions -- unlike
    'knowledge_base_lookup', which only knows a handful of OUR OWN mock
    internal company facts (refund policy, SLA, pricing), this can answer
    almost anything with a Wikipedia article. Requires internet access.
    """
    try:
        result = _wikipedia_tool.invoke({"query": query})
    except Exception as exc:  # network/SSL/API errors -- never crash the agent loop
        return f"Error searching Wikipedia for '{query}': {exc}"
    return result or f"No Wikipedia results found for '{query}'."


# NOTE ON SECURITY MODEL (read this before reusing this tool anywhere else):
# This was originally an ALLOWLIST of read-only command names + shell=False,
# which is the safe pattern for any real/production system: the LLM's
# intentions don't matter, the tool itself enforces what can physically run.
#
# At the user's explicit request, that boundary has been REMOVED below so the
# model has full freedom to run any shell command it decides it needs,
# including pipes/redirects/chaining via a real shell (shell=True). This is
# appropriate ONLY for a local demo/learning sandbox running as your own user
# on your own machine. It is NOT safe to expose this to untrusted input, a
# multi-user service, or anything prompt-injectable from the outside (e.g. a
# tool that reads attacker-controlled web pages) -- a model can be tricked
# into running destructive commands (rm -rf, curl | sh, etc.) and nothing
# here will stop it. If you ever productionize this, bring back the allowlist.
SHELL_COMMAND_TIMEOUT_SECONDS = 10
SHELL_OUTPUT_MAX_CHARS = 4000


@tool
def run_shell_command(command: str) -> str:
    """Run any shell command (full bash/zsh syntax: pipes, redirects, chaining,
    globs, all standard Unix tools like wc/grep/find/awk/sed/xargs) and return
    its combined stdout+stderr.

    This has NO allowlist and NO restrictions on which programs or shell
    features can be used -- you have full freedom to run whatever command is
    needed to answer the question, including multi-step pipelines like
    "ls -la | wc -l" or "find . -name '*.py' | xargs wc -l". There is a
    10-second timeout and output is truncated if extremely long, but
    otherwise this behaves like a normal shell prompt. Use it for file
    system, process, OS, and system-info questions, or anything else where
    running a real command gives a more accurate answer than guessing.

    THIS RUNS ON THE SERVER'S OWN MACHINE WITH THE SERVER'S OWN PERMISSIONS --
    avoid destructive commands (deleting/overwriting files, killing
    processes, modifying system state) unless the user has clearly asked for
    exactly that.
    """
    command = command.strip()
    if not command:
        return "Error: empty command"

    try:
        result = subprocess.run(
            command,
            shell=True,  # intentional: gives the model a real shell (pipes, redirects, globs, etc.)
            capture_output=True,
            text=True,
            timeout=SHELL_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {SHELL_COMMAND_TIMEOUT_SECONDS}s"

    output = (result.stdout or "") + (result.stderr or "")
    output = output.strip() or "(command produced no output)"
    if result.returncode != 0:
        output = f"[exit code {result.returncode}]\n{output}"
    if len(output) > SHELL_OUTPUT_MAX_CHARS:
        output = output[:SHELL_OUTPUT_MAX_CHARS] + "... [truncated]"
    return output


# Full file-system toolkit: use LangChain's PREBUILT ``FileManagementToolkit``
# (langchain_community.agent_toolkits) instead of hand-rolling pathlib-based
# tools. Leaving ``selected_tools`` unset exposes its ENTIRE family of
# ready-made, agent-ready file-system tools -- read_file, write_file,
# list_directory, file_search, copy_file, move_file, file_delete -- so the
# model has the full range of file operations a real coding assistant needs,
# not just a hand-picked subset. Same security-model note as
# run_shell_command applies: ``root_dir=None`` means no path restriction,
# which is appropriate for a local demo/learning sandbox only -- pass a real
# ``root_dir`` to sandbox these to one directory in any less trusted setting.
_file_tools = {t.name: t for t in FileManagementToolkit(root_dir=None).get_tools()}
list_directory = _file_tools["list_directory"]
read_file = _file_tools["read_file"]
write_file = _file_tools["write_file"]
file_search = _file_tools["file_search"]
copy_file = _file_tools["copy_file"]
move_file = _file_tools["move_file"]
file_delete = _file_tools["file_delete"]


@tool
def system_resources() -> str:
    """Report current CPU usage, memory usage, and disk usage as exact numbers.

    Computed directly via the 'psutil' library (no shelling out to 'df',
    'top', or 'uptime' and parsing their text). Use this for questions like
    "how much memory/disk/CPU is being used right now".
    """
    cpu_percent = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return (
        f"CPU usage: {cpu_percent:.1f}%\n"
        f"Memory: {mem.percent:.1f}% used "
        f"({mem.used // (1024 ** 2)} MB used / {mem.total // (1024 ** 2)} MB total)\n"
        f"Disk ('/'): {disk.percent:.1f}% used "
        f"({disk.used // (1024 ** 3)} GB used / {disk.total // (1024 ** 3)} GB total)"
    )


# edit_file has no off-the-shelf LangChain equivalent (the built-in file
# toolkit can read/write/move/copy/delete whole files but has no "replace
# this exact snippet" tool), so it stays hand-written -- it gives the model
# a far less error-prone way to change PART of a file than reconstructing
# the whole content by hand via 'write_file'.


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace an exact snippet of text within an existing file with new text.

    'old_text' must match EXACTLY ONE location in the file (including
    whitespace/indentation) -- if it appears zero times or more than once,
    nothing is changed and an error is returned explaining why, so you can
    adjust 'old_text' (e.g. include a bit more surrounding context) and try
    again. This is much safer than rewriting a whole file via 'write_file',
    since it can't accidentally discard unrelated content you didn't mean to
    touch.

    Use 'read_file' first to see the file's exact current content (including
    whitespace) before choosing 'old_text'. 'path' is resolved relative to the
    server's working directory (the project root) if not absolute.
    """
    try:
        target = Path(path).expanduser()
        if not target.exists():
            return f"Error: path '{path}' does not exist"
        if not target.is_file():
            return f"Error: '{path}' is not a file"
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: '{path}' is not a readable text file (binary content)"
    except OSError as exc:
        return f"Error: could not read '{path}' ({exc})"

    occurrences = content.count(old_text)
    if occurrences == 0:
        return f"Error: old_text not found in '{path}' -- no changes made"
    if occurrences > 1:
        return (
            f"Error: old_text matches {occurrences} locations in '{path}' -- "
            "include more surrounding context to make it unique, no changes made"
        )

    try:
        target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    except OSError as exc:
        return f"Error: could not write '{path}' ({exc})"

    return f"Edited '{path}': replaced 1 occurrence."


AVAILABLE_TOOLS = [
    calculator,
    knowledge_base_lookup,
    wikipedia_search,
    run_shell_command,
    list_directory,
    read_file,
    write_file,
    edit_file,
    file_search,
    copy_file,
    move_file,
    file_delete,
    system_resources,
    rag_search,
    codebase_search,
    readme_search,
    article_search,
]
