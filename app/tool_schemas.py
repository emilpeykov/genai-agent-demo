"""Pydantic validation schemas for each tool's arguments.

Local tool-calling models (especially smaller/quantized ones served over
Ollama) don't always emit arguments that match a tool's expected shape --
e.g. a boolean sent as the literal string ``"False"`` instead of a real
JSON boolean, a required field omitted entirely, or a field misspelled.
Historically ``ReActAgent._execute_tool_call`` just handed whatever dict
the model produced straight to ``tool.invoke(args)`` and let the
underlying exception bubble up into a generic
``"Error running tool 'x': ..."`` observation -- which told the model
THAT something broke, but not precisely WHAT shape was actually expected,
so its retry was often just as wrong.

This module validates (and type-COERCES, e.g. the string "false"/"False"
-> the real boolean ``False``) each tool call's raw arguments through an
explicit pydantic model before the real tool function ever runs. On
failure, the resulting ``pydantic.ValidationError`` is rendered into a
short, precise, actionable observation (exact field name + exact problem)
and fed back into the model's own context as a normal tool observation --
the standard "validate client-side, hand the validation error back so the
model can read its own mistake and correct the tool call on its next
reasoning pass" pattern for reliable tool-calling.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError


class _Calculator(BaseModel):
    expression: str


class _KnowledgeBaseLookup(BaseModel):
    topic: str


class _QueryOnly(BaseModel):
    """Shared shape for the RAG-style tools, which all take a single
    'query' string (rag_search, codebase_search, readme_search,
    article_search, wikipedia_search, web_search, google_search,
    arxiv_search, stackexchange_search)."""

    query: str


class _FetchWebpage(BaseModel):
    url: str


class _CrawlWebsite(BaseModel):
    url: str
    max_pages: int = 5


class _RunShellCommand(BaseModel):
    command: str


class _ListDirectory(BaseModel):
    dir_path: str = "."


class _ReadFile(BaseModel):
    file_path: str


class _WriteFile(BaseModel):
    file_path: str
    text: str
    append: bool = False


class _EditFile(BaseModel):
    path: str
    old_text: str
    new_text: str


class _FileSearch(BaseModel):
    pattern: str
    dir_path: str = "."


class _GrepSearch(BaseModel):
    query: str
    dir_path: str = "."
    include_pattern: str | None = None
    is_regex: bool = False
    max_results: int = 50


class _GetErrors(BaseModel):
    file_paths: list[str]


class _AskUser(BaseModel):
    question: str


class _CopyFile(BaseModel):
    source_path: str
    destination_path: str


class _MoveFile(BaseModel):
    source_path: str
    destination_path: str


class _FileDelete(BaseModel):
    file_path: str


class _NoArgs(BaseModel):
    """Shape for tools that take no arguments at all (system_resources)."""

    model_config = {"extra": "forbid"}


# Maps each real tool name (from app.tools.AVAILABLE_TOOLS) to the pydantic
# model describing its expected arguments. Kept as an explicit dict (rather
# than deriving schemas generically from each tool's own ``.args`` JSON
# schema) so every tool's validation rules are visible and editable in one
# place, and so adding a genuinely NEW constraint (e.g. "path must be
# absolute") later is just normal pydantic, not schema-generation code.
_SCHEMAS_BY_TOOL_NAME: dict[str, type[BaseModel]] = {
    "calculator": _Calculator,
    "knowledge_base_lookup": _KnowledgeBaseLookup,
    "wikipedia_search": _QueryOnly,
    "web_search": _QueryOnly,
    "google_search": _QueryOnly,
    "arxiv_search": _QueryOnly,
    "stackexchange_search": _QueryOnly,
    "fetch_webpage": _FetchWebpage,
    "crawl_website": _CrawlWebsite,
    "run_shell_command": _RunShellCommand,
    "list_directory": _ListDirectory,
    "read_file": _ReadFile,
    "write_file": _WriteFile,
    "edit_file": _EditFile,
    "file_search": _FileSearch,
    "grep_search": _GrepSearch,
    "get_errors": _GetErrors,
    "copy_file": _CopyFile,
    "move_file": _MoveFile,
    "file_delete": _FileDelete,
    "system_resources": _NoArgs,
    "ask_user": _AskUser,
    "rag_search": _QueryOnly,
    "codebase_search": _QueryOnly,
    "readme_search": _QueryOnly,
    "article_search": _QueryOnly,
}


def _format_validation_error(tool_name: str, exc: ValidationError) -> str:
    """Render a pydantic ``ValidationError`` as a short, model-readable
    observation -- which field(s), what was wrong, and what to do next --
    rather than pydantic's full multi-line error dump, which is mostly
    noise to a model trying to self-correct a tool call."""
    problems = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(top-level)"
        problems.append(f"'{loc}': {err['msg']}")
    problem_text = "; ".join(problems)
    return (
        f"Error: invalid arguments for tool '{tool_name}' -- {problem_text}. "
        "Re-call this SAME tool again, in your very next turn, with "
        "corrected arguments matching its documented parameters exactly -- "
        "do not give up, apologize, or answer the user without the real "
        "tool result."
    )


def validate_tool_args(
    tool_name: str, args: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and type-coerce ``args`` for ``tool_name``.

    Returns ``(coerced_args, None)`` on success -- ``coerced_args`` has the
    same shape the real tool function expects, with simple type coercions
    (e.g. the string ``"False"`` -> the real boolean ``False``) already
    applied by pydantic -- or ``(None, error_observation)`` on failure,
    where ``error_observation`` is ready to hand straight back to the model
    as the tool call's observation.

    A tool name with no registered schema (there shouldn't be any, since
    this is meant to cover every entry in ``app.tools.AVAILABLE_TOOLS``) is
    passed through UNVALIDATED rather than rejected, so a newly added tool
    that hasn't been given a schema here yet still works -- it just skips
    this extra safety net instead of breaking.
    """
    schema = _SCHEMAS_BY_TOOL_NAME.get(tool_name)
    if schema is None:
        return args, None
    try:
        validated = schema.model_validate(args)
    except ValidationError as exc:
        return None, _format_validation_error(tool_name, exc)
    return validated.model_dump(), None

