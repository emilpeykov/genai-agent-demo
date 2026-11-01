"""Tools exposed to the agent, defined with LangChain's ``@tool`` decorator.

These illustrate the "agent frameworks" requirement: the agent can decide,
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
# knowledge source, this reuses LangChain's PREBUILT ``WikipediaAPIWrapper``
# (from langchain_community.utilities) -- the "use a predefined tool/library"
# alternative to writing a custom web-search tool from scratch. LangChain
# ships many of these ready-made integrations (see module docstring below for
# more); this is a concrete example rather than just a description of one.
_wikipedia = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)


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
        result = _wikipedia.run(query)
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


# These next two tools demonstrate the alternative to "wrap a shell command":
# use a native Python library (pathlib, psutil) instead of subprocess at all.
# No shlex parsing, no shell metacharacters to worry about, and the model
# gets exact structured numbers instead of having to parse free-text output
# or reach for a Unix pipe idiom (ls | wc -l) that this project doesn't
# support. Any Python library can be turned into a tool this way: write a
# plain function, decorate it with @tool, give it a clear docstring.

DIRECTORY_LISTING_MAX_ENTRIES = 200


@tool
def list_directory(path: str = ".") -> str:
    """List files and subdirectories directly under a given path, with a count.

    Returns the total number of files, the total number of subdirectories,
    and their names -- computed directly in Python (pathlib), not by running
    'ls' and parsing its text. Use this whenever asked to count, list, or
    check for the existence of files/folders -- it always gives an exact
    count, so there is never a need to pipe output through 'wc' or 'grep'.

    'path' defaults to the current directory ('.') if not given. Relative
    paths are resolved from the server's working directory.
    """
    try:
        target = Path(path).expanduser()
        if not target.exists():
            return f"Error: path '{path}' does not exist"
        if not target.is_dir():
            return f"Error: '{path}' is not a directory"
        entries = sorted(target.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        return f"Error: could not list '{path}' ({exc})"

    files = [e.name for e in entries if e.is_file()]
    dirs = [e.name for e in entries if e.is_dir()]
    shown_files = files[:DIRECTORY_LISTING_MAX_ENTRIES]
    shown_dirs = dirs[:DIRECTORY_LISTING_MAX_ENTRIES]
    summary = (
        f"{len(files)} file(s), {len(dirs)} subdirectory(ies) in '{path}'.\n"
        f"Files: {', '.join(shown_files) if shown_files else '(none)'}\n"
        f"Subdirectories: {', '.join(shown_dirs) if shown_dirs else '(none)'}"
    )
    return summary


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


# Why this tool exists: 'codebase_search' does SEMANTIC (embedding similarity)
# search -- great for fuzzy questions like "how is X implemented?", but
# unreliable when the user names a SPECIFIC file (e.g. "check the README").
# A vague/short query can embed closer to an unrelated chunk than to the
# actual file being asked about (observed: a query literally containing the
# word "readme" top-matched app/codebase_rag.py instead of README.md).
# read_file sidesteps retrieval quality entirely for this common case: give
# it an exact path, get back the exact, complete, ground-truth content.
READ_FILE_MAX_CHARS = 8000


@tool
def read_file(path: str) -> str:
    """Read and return the exact, full contents of a specific file by path.

    Use this whenever the user names a SPECIFIC file (e.g. "check the
    README", "what does app/agent.py do", "read config.py") -- this gives
    you the exact real content, instead of 'codebase_search', which does
    fuzzy semantic search and can match the wrong file/chunk for short or
    oddly-phrased queries. Prefer 'read_file' over 'codebase_search' any
    time you already know (or can guess) the file's path or name.

    'path' is resolved relative to the server's working directory (the
    project root) if not absolute. Binary files and files over roughly
    8000 characters are rejected/truncated to keep responses manageable --
    use 'list_directory' first if you need to find the right path.
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

    if len(content) > READ_FILE_MAX_CHARS:
        content = content[:READ_FILE_MAX_CHARS] + "\n... [truncated]"
    return content


AVAILABLE_TOOLS = [
    calculator,
    knowledge_base_lookup,
    wikipedia_search,
    run_shell_command,
    list_directory,
    read_file,
    system_resources,
    rag_search,
    codebase_search,
    readme_search,
    article_search,
]
