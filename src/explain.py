def build_facts(recommendation, budget, runtime_hours, deadline_hours, risk_profile):
    """
    Converts raw recommendation output + user inputs into a 'facts' dictionary
    that the rule engine will reason over.
    """
    facts = {
        "interruption_probability": recommendation["interruption_probability"],  # %
        "completion_probability": recommendation["completion_probability"],      # %
        "expected_cost": recommendation["expected_cost"],
        "budget": budget,
        "risk_level": recommendation["risk_level"],
        "risk_profile": risk_profile,
        "runtime_hours": runtime_hours,
        "deadline_hours": deadline_hours,
        "estimated_savings_pct": recommendation["estimated_savings_pct"],
    }

    # Derived fact: how much slack time is left before deadline
    facts["deadline_slack"] = deadline_hours - runtime_hours

    # Derived fact: how close to budget the expected cost is
    facts["budget_headroom_pct"] = (
        ((budget - recommendation["expected_cost"]) / budget) * 100
        if budget > 0 else 0
    )

    return facts
def apply_rules(facts):
    """
    Forward chaining: checks each rule's condition against current facts.
    If true, the rule 'fires' and its conclusion is added to the explanation list.
    """
    explanations = []

    # Rule 1: High interruption risk
    if facts["interruption_probability"] > 40:
        explanations.append(
            "Interruption risk is high because the predicted market price is "
            "close to or above your bid — consider raising your bid if the job is critical."
        )

    # Rule 2: Low interruption risk
    elif facts["interruption_probability"] < 15:
        explanations.append(
            "Interruption risk is low, meaning your bid comfortably covers "
            "expected market price fluctuations."
        )

    # Rule 3: Tight deadline
    if facts["deadline_slack"] < 2:
        explanations.append(
            "Your deadline is tight relative to the job runtime, so a higher bid "
            "was favored to reduce the chance of a costly restart."
        )

    # Rule 4: Comfortable deadline slack
    elif facts["deadline_slack"] >= 5:
        explanations.append(
            "You have comfortable slack before your deadline, allowing a lower, "
            "more cost-conscious bid."
        )

    # Rule 5: Budget is tight
    if facts["budget_headroom_pct"] < 10:
        explanations.append(
            "Expected cost is very close to your budget limit — there is little "
            "room to raise the bid further without exceeding it."
        )

    # Rule 6: Budget has healthy headroom
    elif facts["budget_headroom_pct"] > 40:
        explanations.append(
            "Your budget has healthy headroom, so the recommendation could "
            "afford a higher bid if you want lower interruption risk."
        )

        # Rule 7: Risk profile influence - Conservative (only note if it actually mattered)
    if facts["risk_profile"] == "Conservative" and facts["interruption_probability"] > 15:
        explanations.append(
            "Because you selected a Conservative risk profile, the system "
            "prioritized minimizing interruption risk over minimizing cost."
        )

    # Rule 8: Risk profile influence - Aggressive (only note if risk is meaningfully elevated)
    if facts["risk_profile"] == "Aggressive" and facts["interruption_probability"] > 15:
        explanations.append(
            "Because you selected an Aggressive risk profile, the system "
            "accepted a higher interruption risk in exchange for cost savings."
        )
    elif facts["risk_profile"] == "Aggressive" and facts["interruption_probability"] <= 15:
        explanations.append(
            "Even with an Aggressive risk profile, market conditions were "
            "favorable enough that this bid still keeps interruption risk low."
        )

    # Rule 9: Good savings achieved
    if facts["estimated_savings_pct"] > 50:
        explanations.append(
            f"This bid achieves an estimated {facts['estimated_savings_pct']:.1f}% "
            "savings compared to on-demand pricing."
        )

    # Fallback rule: nothing strong triggered
    if not explanations:
        explanations.append(
            "This bid represents a balanced trade-off between cost and "
            "interruption risk given your inputs."
        )

    return explanations
def get_explanation(recommendation, budget, runtime_hours, deadline_hours, risk_profile):
    """
    Main entry point: builds facts, applies rules, returns explanation text.
    """
    facts = build_facts(recommendation, budget, runtime_hours, deadline_hours, risk_profile)
    explanations = apply_rules(facts)

    # Join into a single readable paragraph
    explanation_text = " ".join(explanations)

    return explanation_text, facts
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from search import get_recommendation

    recommendation = get_recommendation(
        instance_type="m4.xlarge", availability_zone="us-east-1a",
        operating_system="Linux/UNIX", hour=14, day_of_week=2,
        runtime_hours=4, budget=5.0, risk_profile="Conservative"
    )

    print("\n--- Recommendation ---")
    print(recommendation)

    explanation_text, facts = get_explanation(
        recommendation=recommendation,
        budget=5.0,
        runtime_hours=4,
        deadline_hours=6,
        risk_profile="Aggressive"
    )

    print("\n--- Explanation ---")
    print(explanation_text)

    print("\n--- Facts used ---")
    print(facts)