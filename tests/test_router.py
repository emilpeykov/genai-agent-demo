from app.router import classify_complexity


def test_short_prompt_routes_to_fast_tier():
    decision = classify_complexity("What's the capital of France?")
    assert decision.tier == "fast"


def test_moderate_prompt_routes_to_balanced_tier():
    decision = classify_complexity(
        "Summarize the key differences between REST and gRPC for a non-technical stakeholder."
    )
    assert decision.tier == "balanced"


def test_complex_prompt_routes_to_powerful_tier():
    decision = classify_complexity(
        "Design a step-by-step algorithm to optimize our model-routing strategy "
        "given latency and cost trade-offs across three model tiers."
    )
    assert decision.tier == "powerful"


def test_long_prompt_escalates_even_without_keywords():
    long_prompt = "word " * 150
    decision = classify_complexity(long_prompt)
    assert decision.tier == "powerful"
