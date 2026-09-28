def build_facts(recommendation, budget, runtime_hours, deadline_hours, risk_profile):
    """Packages recommendation output + user inputs into a facts dictionary."""
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

    # Derived facts
    facts["deadline_slack"] = deadline_hours - runtime_hours
    facts["budget_headroom_pct"] = (
        ((budget - recommendation["expected_cost"]) / budget) * 100
        if budget > 0 else 0
    )
    return facts


# Rule format: (name, condition, derived facts, message or None)
RULES = [
    ("R1", lambda f: f["interruption_probability"] > 40,
     {"risk_high": True}, None),

    ("R2", lambda f: f["interruption_probability"] < 15,
     {"risk_low": True},
     "Interruption risk is low ({interruption_probability}%): the bid sits above most historical prices for this instance."),
    ("R14", lambda f: 15 <= f["interruption_probability"] <= 40,
     {"risk_medium": True},
     "Interruption risk is moderate ({interruption_probability}%): the job has a real chance of needing a restart, which is the trade-off for the lower bid."),

    ("R3", lambda f: f["deadline_slack"] < 2,
     {"time_pressure": True}, None),

    ("R4", lambda f: f["deadline_slack"] >= 5,
     {"time_relaxed": True},
     "You have {deadline_slack} hours of slack before the deadline, so a cost-conscious bid is acceptable."),

    ("R5", lambda f: f["budget_headroom_pct"] < 10,
     {"budget_tight": True},
     "Expected spend is within 10% of your budget, so there is little room to raise the bid."),

    ("R6", lambda f: f["budget_headroom_pct"] > 40,
     {"budget_healthy": True},
     "Your budget has healthy headroom."),

    # Chained rules: these use facts derived by earlier rules
    ("R7", lambda f: f.get("risk_high") and f.get("time_pressure"),
     {"advise_raise_bid": True},
     "High interruption risk with a tight deadline: a restart would likely miss the deadline."),

    ("R8", lambda f: f.get("risk_high") and f.get("budget_tight"),
     {"tradeoff_conflict": True},
     "The bid cannot be raised without exceeding the budget: consider a longer deadline or a bigger budget."),

    ("R9", lambda f: f.get("advise_raise_bid") and not f.get("budget_tight"),
     {"raise_bid_possible": True},
     "Your budget allows a higher bid - consider raising it to reduce interruption risk."),

    ("R10", lambda f: f["risk_profile"] == "Conservative",
     {"profile_known": True},
     "Conservative profile: the search assumed prices could spike to their 99th percentile (limited by your budget and the on-demand cap), so it favored a high bid."),

    ("R11", lambda f: f["risk_profile"] == "Balanced",
     {"profile_known": True},
     "Balanced profile: the search assumed prices up to the 97th percentile, balancing cost against interruption risk."),

    ("R12", lambda f: f["risk_profile"] == "Aggressive",
     {"profile_known": True},
     "Aggressive profile: the search assumed only the 90th-percentile price, accepting more interruption risk to save cost."),

    ("R13", lambda f: f["estimated_savings_pct"] > 30,
     {"good_savings": True},
     "This bid gives an estimated {estimated_savings_pct:.1f}% saving versus on-demand pricing."),
]


def forward_chain(facts):
    """Forward chaining: keep firing rules until no new rule fires."""
    facts = dict(facts)
    fired = []
    explanations = []
    changed = True
    while changed:
        changed = False
        for name, condition, derived, message in RULES:
            if name in fired:
                continue
            if condition(facts):
                fired.append(name)
                facts.update(derived)
                if message:
                    explanations.append(message.format(**facts))
                changed = True
    facts["rules_fired"] = fired
    return explanations, facts


def get_explanation(recommendation, budget, runtime_hours, deadline_hours, risk_profile):
    facts = build_facts(recommendation, budget, runtime_hours, deadline_hours, risk_profile)
    explanations, facts = forward_chain(facts)
    if not explanations:
        explanations = ["This bid is a balanced trade-off between cost and interruption risk for your inputs."]
    return " ".join(explanations), facts


if __name__ == "__main__":
    from search import get_recommendation

    for profile in ["Conservative", "Balanced", "Aggressive"]:
        rec = get_recommendation(
            "m4.xlarge", "us-east-1a", "Linux/UNIX",
            hour=14, day_of_week=2, runtime_hours=4,
            budget=11.0, risk_profile=profile, deadline_hours=6
        )
        text, facts = get_explanation(rec, 11.0, 4, 6, profile)
        print(f"\n=== {profile} ===")
        print("Bid:", rec["recommended_bid"], "| Completion:", rec["completion_probability"], "%")
        print("Explanation:", text)
        print("Rules fired:", facts["rules_fired"])