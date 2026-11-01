from app.prompts import render_assistant_prompt, render_react_system_prompt


def test_render_assistant_prompt_includes_company_name_and_user_request():
    rendered = render_assistant_prompt("What is the refund policy?", company_name="TestCo")

    assert "TestCo" in rendered
    assert "What is the refund policy?" in rendered
    assert "User request:" in rendered


def test_render_assistant_prompt_includes_todays_date():
    from datetime import date

    rendered = render_assistant_prompt("hello")

    assert date.today().isoformat() in rendered


def test_render_assistant_prompt_uses_default_company_name_when_not_given():
    rendered = render_assistant_prompt("hello")

    assert "Acme GenAI Demo Co." in rendered


def test_render_react_system_prompt_lists_all_given_tool_names():
    rendered = render_react_system_prompt(["calculator", "run_shell_command"], company_name="TestCo")

    assert "calculator" in rendered
    assert "run_shell_command" in rendered
    assert "TestCo" in rendered


def test_render_react_system_prompt_reflects_the_real_tool_registry():
    # Regression guard: the system prompt must be derived from the ACTUAL
    # tool list (app.tools.AVAILABLE_TOOLS), not a hand-maintained copy that
    # could silently drift out of sync as tools are added/removed.
    from app.tools import AVAILABLE_TOOLS

    rendered = render_react_system_prompt([t.name for t in AVAILABLE_TOOLS])

    for tool in AVAILABLE_TOOLS:
        assert tool.name in rendered


def test_render_react_system_prompt_tells_model_to_prefer_read_file_for_named_files():
    # Regression test: a query like "check the README" was observed to route
    # through codebase_search's fuzzy semantic matching and confidently
    # return the WRONG file (app/codebase_rag.py instead of README.md). The
    # prompt must steer the model to use read_file/readme_search instead.
    rendered = render_react_system_prompt(["read_file", "readme_search", "codebase_search"])

    assert "read_file" in rendered
    assert "readme_search" in rendered
