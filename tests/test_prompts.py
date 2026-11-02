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


def test_render_react_system_prompt_tells_model_memory_does_not_need_tools():
    # Regression test: asked to "remember" a fact, a model was observed
    # calling write_file to persist it (sometimes even emitting the tool
    # call as raw JSON text instead of a real structured tool call). The
    # prompt must make clear that conversation history IS the memory, and
    # that tool calls must go through the real tool-calling mechanism.
    rendered = render_react_system_prompt(["write_file"])

    assert "Do NOT call 'write_file'" in rendered
    assert "never write out a JSON object" in rendered


def test_render_react_system_prompt_tells_model_it_can_answer_project_meta_questions():
    # Regression test: asked "what is the project root dir name?", a model
    # flatly refused with "I don't have the necessary tools..." even though
    # run_shell_command/list_directory can answer this. The prompt must make
    # clear the model DOES have the tools and should use them, not refuse.
    rendered = render_react_system_prompt(["run_shell_command", "list_directory"])

    assert "PROJECT META-QUESTIONS" in rendered
    assert "run_shell_command" in rendered
    assert "list_directory" in rendered
    assert "never say you lack the necessary tools" in rendered


def test_render_react_system_prompt_tells_model_to_look_up_commands_before_running_them():
    # Regression test: asked to "run the curl test command shown in the
    # README", a model fabricated an UNRELATED command
    # (curl https://example.com | grep 'Hello World') instead of looking up
    # the real one via readme_search/read_file -- then reported on that
    # made-up command's result as if it answered the question. The prompt
    # must forbid guessing/inventing commands and instead require looking
    # up the real text first.
    rendered = render_react_system_prompt(["run_shell_command", "readme_search", "read_file"])

    assert "COMMAND LOOKUP" in rendered
    assert "Do NOT guess, invent, or " in rendered
    assert "NOT a sign the " in rendered

