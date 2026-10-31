from app.agent import Agent
from app.providers import MockLLMProvider


def test_agent_routes_arithmetic_to_calculator_tool():
    agent = Agent(provider=MockLLMProvider(seed=1))
    result = agent.handle("2 + 2 * 3")
    assert result.tool_used == "calculator"
    assert result.answer == "8"
    assert result.estimated_cost_usd == 0.0


def test_agent_routes_kb_question_to_tool():
    agent = Agent(provider=MockLLMProvider(seed=1))
    result = agent.handle("What is our refund policy?")
    assert result.tool_used == "knowledge_base_lookup"
    assert "business days" in result.answer


def test_agent_routes_os_info_question_to_shell_tool():
    agent = Agent(provider=MockLLMProvider(seed=1))
    result = agent.handle("What is the current working directory?")
    assert result.tool_used == "run_shell_command"
    assert result.estimated_cost_usd == 0.0
    assert "/" in result.answer


def test_agent_routes_general_question_to_llm():
    agent = Agent(provider=MockLLMProvider(seed=1))
    result = agent.handle("What's the capital of France?")
    assert result.tool_used is None
    assert result.tier == "fast"
    assert result.model is not None


def test_agent_routes_complex_question_to_powerful_tier():
    agent = Agent(provider=MockLLMProvider(seed=1))
    result = agent.handle(
        "Design a step-by-step algorithm to optimize our model-routing strategy "
        "given latency and cost trade-offs."
    )
    assert result.tier == "powerful"
