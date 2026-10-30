from app.tools import calculator, knowledge_base_lookup


def test_calculator_tool_basic_arithmetic():
    assert calculator.invoke({"expression": "2 + 2"}) == "4"


def test_calculator_tool_handles_precedence():
    assert calculator.invoke({"expression": "2 + 3 * 4"}) == "14"


def test_calculator_tool_rejects_unsafe_expression():
    assert calculator.invoke({"expression": "__import__('os').system('ls')"}) == (
        "Error: could not evaluate expression"
    )


def test_knowledge_base_lookup_known_topic():
    result = knowledge_base_lookup.invoke({"topic": "refund policy"})
    assert "5-7 business days" in result


def test_knowledge_base_lookup_unknown_topic():
    result = knowledge_base_lookup.invoke({"topic": "unknown thing"})
    assert "No internal knowledge found" in result
