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


def test_render_react_system_prompt_distinguishes_test_command_from_setup_commands():
    # Regression test: asked to read the README and "execute the test command
    # described there", the model ran "uvicorn app.main:app --reload" (a
    # SETUP/run step from the "Running locally" section) instead of the
    # actual "Try it:" curl example -- which failed with "Address already in
    # use" (since a server was already running and answering the request),
    # wasting a tool-call round and contributing to "the agent just stopped,
    # no answer" when the round cap was then hit. The prompt must make clear
    # the "test command" means the example request, not the setup/run steps,
    # and that starting another server is both unnecessary and will fail.
    rendered = render_react_system_prompt(["run_shell_command", "readme_search", "read_file"])

    assert "Address already in use" in rendered
    assert "already running inside a live server process" in rendered
    assert "Do NOT run setup commands" in rendered


def test_render_react_system_prompt_forbids_describing_a_plan_without_acting():
    # Regression test: asked to "check the current dir, read the README, and
    # execute the test command described there", the model wrote a long,
    # helpful-sounding plan ("I'll start by setting up... Let's proceed with
    # executing these steps...") but never actually issued ANY tool call in
    # that turn -- the loop then correctly treated this as the model's final
    # answer (no tool_calls were requested), so nothing was ever executed.
    # The prompt must explicitly forbid describing-without-acting.
    rendered = render_react_system_prompt(["run_shell_command", "readme_search", "read_file"])

    assert "NEVER" in rendered
    assert "describe a plan to call a tool and then stop without calling it" in rendered
    assert "actually issue the real tool call in this SAME turn" in rendered


def test_render_react_system_prompt_tells_model_to_use_real_tool_output_verbatim():
    # Regression test: asked to "get the current memory usage and save it to
    # a file", qwen3:14b correctly called system_resources, reported the real
    # numbers in its chat answer, but then called write_file with a
    # fabricated placeholder string (e.g. "Memory usage: 1234567890 bytes")
    # instead of the real observation text -- so the FILE ON DISK ended up
    # wrong even though the chat answer looked correct. The prompt must
    # explicitly require copying a prior tool's real output verbatim into a
    # later tool call's arguments, not paraphrasing/inventing a placeholder.
    rendered = render_react_system_prompt(["system_resources", "write_file"])

    assert "USE REAL TOOL OUTPUT VERBATIM" in rendered
    assert "character for character" in rendered
    assert "Do NOT invent, paraphrase, " in rendered


def test_render_react_system_prompt_tells_model_to_escalate_across_search_tools():
    # Regression test: asked to research a topic, the model called
    # 'web_search' once, got a thin/short result, and gave a final answer
    # without ever trying 'google_search'/'wikipedia_search'/'fetch_webpage'
    # to get a fuller answer. The prompt must require escalating through
    # the other search tools (and fetching a result URL's full page text)
    # before settling for one shallow search result.
    rendered = render_react_system_prompt(
        ["web_search", "google_search", "wikipedia_search", "fetch_webpage"]
    )

    assert "SEARCH ESCALATION" in rendered
    assert "NOT a sign the information doesn't exist" in rendered
    assert "fetch_webpage" in rendered
    assert "do not give up after a single thin 'web_search' call" in rendered


def test_render_react_system_prompt_tells_model_to_always_state_exact_file_paths():
    # Regression test: asked to create a file in one turn, then "read back
    # that file" in a later turn, qwen3:14b had no memory of the exact path
    # it had just used -- it had to fall back on 'find the most recently
    # modified file' instead of just knowing it. Root cause: ReActAgent.handle()
    # only replays each past turn's plain ANSWER TEXT into history, never the
    # real tool_calls/arguments/observations behind it -- so if the model's
    # own answer was vague ("I've saved it for you") instead of stating the
    # real path, that path is permanently lost for later turns. The prompt
    # must require the final answer to always restate the exact file path(s)
    # involved in any file operation.
    rendered = render_react_system_prompt(["write_file", "read_file", "edit_file"])

    assert "ALWAYS STATE EXACT FILE PATHS IN YOUR ANSWER" in rendered
    assert "only the plain TEXT of your own past final answers is" in rendered
    assert "MUST explicitly restate the exact, full path(s) involved" in rendered


