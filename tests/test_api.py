from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_warm_up_vector_indexes_is_a_noop_under_pytest():
    # Regression test: _warm_up_vector_indexes must never make a real
    # network call to Ollama during the test suite (PYTEST_CURRENT_TEST is
    # always set by pytest itself) -- it should return immediately without
    # starting a background thread.
    import threading

    before = threading.active_count()
    main_module._warm_up_vector_indexes()
    assert threading.active_count() == before


def test_chat_endpoint_tool_path():
    response = client.post("/chat", json={"prompt": "10 * 4"})
    assert response.status_code == 200
    body = response.json()
    assert body["tool_used"] == "calculator"
    assert body["answer"] == "40"


def test_chat_endpoint_llm_path():
    response = client.post("/chat", json={"prompt": "Tell me a fun fact."})
    assert response.status_code == 200
    body = response.json()
    assert body["tier"] in {"fast", "balanced", "powerful"}
    assert "answer" in body


def test_chat_endpoint_rejects_empty_prompt():
    response = client.post("/chat", json={"prompt": ""})
    assert response.status_code == 422


def test_chat_lcel_endpoint_returns_answer():
    response = client.post("/chat/lcel", json={"prompt": "Tell me a fun fact."})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["answer"], str)
    assert body["answer"]


def test_chat_lcel_tools_endpoint_returns_answer(monkeypatch):
    # Avoid requiring a real Ollama server in CI -- inject a fake chain with
    # the same Runnable interface the real one would have.
    class _FakeChain:
        def invoke(self, inputs: dict) -> str:
            return f"fake tool-using answer for: {inputs['user_request']}"

    monkeypatch.setattr(main_module, "_lcel_tool_chain", _FakeChain())

    response = client.post("/chat/lcel/tools", json={"prompt": "What is 2 + 2?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fake tool-using answer for: What is 2 + 2?"


def test_chat_lcel_tools_memory_endpoint_passes_through_session_id(monkeypatch):
    # Avoid requiring a real Ollama server in CI -- inject a fake chain with
    # the same RunnableWithMessageHistory-like interface the real one would
    # have (an ``invoke(inputs, config=...)`` call).
    class _FakeMemoryChain:
        def __init__(self):
            self.calls: list[tuple[dict, dict]] = []

        def invoke(self, inputs: dict, config: dict) -> str:
            self.calls.append((inputs, config))
            return f"fake memory answer for: {inputs['user_request']}"

    fake_chain = _FakeMemoryChain()
    monkeypatch.setattr(main_module, "_lcel_tool_chain_with_memory", fake_chain)

    response = client.post(
        "/chat/lcel/tools/memory",
        json={"prompt": "My name is Ada.", "session_id": "test-session-xyz"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fake memory answer for: My name is Ada."
    assert fake_chain.calls[0][1] == {"configurable": {"session_id": "test-session-xyz"}}


def test_chat_lcel_tools_memory_endpoint_defaults_session_id(monkeypatch):
    class _FakeMemoryChain:
        def invoke(self, inputs: dict, config: dict) -> str:
            assert config == {"configurable": {"session_id": "default"}}
            return "ok"

    monkeypatch.setattr(main_module, "_lcel_tool_chain_with_memory", _FakeMemoryChain())

    response = client.post("/chat/lcel/tools/memory", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.json()["answer"] == "ok"


def test_chat_lcel_stream_endpoint_streams_chunks(monkeypatch):
    class _FakeStreamingChain:
        def stream(self, inputs: dict):
            yield f"Hello, {inputs['user_request']}"
            yield "!"

    monkeypatch.setattr(
        main_module, "build_assistant_chain_streaming", lambda: _FakeStreamingChain()
    )

    response = client.post("/chat/lcel/stream", json={"prompt": "world"})

    assert response.status_code == 200
    assert response.text == "Hello, world!"


def test_chat_lcel_tools_stream_endpoint_streams_chunks(monkeypatch):
    class _FakeStreamingChain:
        def stream(self, inputs: dict):
            yield "fake "
            yield f"streamed answer for: {inputs['user_request']}"

    monkeypatch.setattr(main_module, "_lcel_tool_chain_streaming", _FakeStreamingChain())

    response = client.post("/chat/lcel/tools/stream", json={"prompt": "What is 2 + 2?"})

    assert response.status_code == 200
    assert response.text == "fake streamed answer for: What is 2 + 2?"


def test_chat_lcel_tools_memory_stream_endpoint_passes_through_session_id(monkeypatch):
    class _FakeStreamingMemoryChain:
        def __init__(self):
            self.calls: list[tuple[dict, dict]] = []

        def stream(self, inputs: dict, config: dict):
            self.calls.append((inputs, config))
            yield f"fake memory stream for: {inputs['user_request']}"

    fake_chain = _FakeStreamingMemoryChain()
    monkeypatch.setattr(main_module, "_lcel_tool_chain_streaming_with_memory", fake_chain)

    response = client.post(
        "/chat/lcel/tools/memory/stream",
        json={"prompt": "My name is Ada.", "session_id": "test-session-xyz"},
    )

    assert response.status_code == 200
    assert response.text == "fake memory stream for: My name is Ada."
    assert fake_chain.calls[0][1] == {"configurable": {"session_id": "test-session-xyz"}}


def test_chat_reasoning_endpoint_returns_reasoning_and_answer(monkeypatch):
    from app.reasoning import ReasoningResult

    def _fake_generate_with_reasoning(prompt: str):
        return ReasoningResult(
            reasoning=f"thinking about: {prompt}",
            answer="42",
            model="deepseek-r1:14b",
            latency_ms=123,
        )

    monkeypatch.setattr(main_module, "generate_with_reasoning", _fake_generate_with_reasoning)

    response = client.post("/chat/reasoning", json={"prompt": "What is 6*7?"})

    assert response.status_code == 200
    body = response.json()
    assert body["reasoning"] == "thinking about: What is 6*7?"
    assert body["answer"] == "42"
    assert body["model"] == "deepseek-r1:14b"
    assert body["latency_ms"] == 123


def test_chat_reasoning_stream_endpoint_streams_ndjson(monkeypatch):
    def _fake_stream_with_reasoning(prompt: str):
        yield "reasoning", "first, I "
        yield "reasoning", "think about it."
        yield "answer", "The answer is 42."

    monkeypatch.setattr(main_module, "stream_with_reasoning", _fake_stream_with_reasoning)

    response = client.post("/chat/reasoning/stream", json={"prompt": "What is 6*7?"})

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line]
    parsed = [main_module.json.loads(line) for line in lines]
    assert parsed == [
        {"kind": "reasoning", "text": "first, I "},
        {"kind": "reasoning", "text": "think about it."},
        {"kind": "answer", "text": "The answer is 42."},
    ]


def test_agent_react_reasoning_endpoint_returns_steps_and_reasoning(monkeypatch):
    from app.reasoning_react_agent import ReasoningReActResult, ReasoningReActStep

    def _fake_handle(prompt: str, history=None):
        return ReasoningReActResult(
            answer="84",
            reasoning="the observation confirms it",
            steps=[
                ReasoningReActStep(
                    tool_name="calculator",
                    tool_args={"expression": "12 * 7"},
                    observation="84",
                    reasoning="I should use the calculator",
                )
            ],
            iterations=2,
            model="deepseek-r1:14b",
            hit_iteration_limit=False,
        )

    monkeypatch.setattr(main_module._reasoning_react_agent, "handle", _fake_handle)

    response = client.post("/agent/react/reasoning", json={"prompt": "What's 12 times 7?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "84"
    assert body["reasoning"] == "the observation confirms it"


def test_agent_react_reasoning_langgraph_endpoint_returns_steps_and_reasoning(monkeypatch):
    from app.reasoning_react_agent import ReasoningReActResult, ReasoningReActStep

    def _fake_handle(prompt: str, history=None):
        return ReasoningReActResult(
            answer="84",
            reasoning="the observation confirms it",
            steps=[
                ReasoningReActStep(
                    tool_name="calculator",
                    tool_args={"expression": "12 * 7"},
                    observation="84",
                    reasoning="I should use the calculator",
                )
            ],
            iterations=2,
            model="deepseek-r1:14b",
            hit_iteration_limit=False,
        )

    monkeypatch.setattr(main_module._reasoning_langgraph_agent, "handle", _fake_handle)

    response = client.post(
        "/agent/react/reasoning/langgraph", json={"prompt": "What's 12 times 7?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "84"
    assert body["reasoning"] == "the observation confirms it"
    assert body["steps"][0]["tool_name"] == "calculator"
    assert body["steps"][0]["tool_name"] == "calculator"
    assert body["steps"][0]["observation"] == "84"
    assert body["model"] == "deepseek-r1:14b"
    assert body["hit_iteration_limit"] is False

