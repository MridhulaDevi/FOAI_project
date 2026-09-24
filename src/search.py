import joblib
import numpy as np
import pandas as pd

# ---- Load trained price model ----
model_bundle = joblib.load("src/utils/price_model.pkl")
model = model_bundle["model"]
feature_cols = model_bundle["feature_cols"]
residual_std = model_bundle["residual_std"]

print("Model loaded. Residual std:", residual_std)
print("Feature columns:", feature_cols)
def get_price_prediction(hour, day_of_week, instance_type, availability_zone, operating_system):
    """
    Builds a single-row feature vector matching training columns,
    returns predicted price and uncertainty (residual_std).
    """
    row = {col: 0 for col in feature_cols}

    if "Hour" in row:
        row["Hour"] = hour
    if "DayOfWeek" in row:
        row["DayOfWeek"] = day_of_week

    # Set relevant one-hot columns to 1 if they exist
    for col in feature_cols:
        if instance_type in col and "InstanceType" in col:
            row[col] = 1
        if availability_zone in col and "AvailabilityZone" in col:
            row[col] = 1
        if operating_system in col and "OperatingSystem" in col:
            row[col] = 1

    X = pd.DataFrame([row])[feature_cols]
    predicted_price = model.predict(X)[0]

    return predicted_price, residual_std


def get_interruption_probability(bid, predicted_price, residual_std, runtime_hours):
    """
    Estimate probability that actual price exceeds bid in a single hour,
    using predicted price +/- residual_std as an approximate normal spread.
    Then extend to full job runtime.
    """
    if residual_std == 0:
        residual_std = 0.01  # avoid divide-by-zero

    # z-score: how many std devs away is the bid from predicted price
    z = (bid - predicted_price) / residual_std

    # Approximate probability price stays BELOW bid using normal CDF
    from scipy.stats import norm
    p_price_below_bid = norm.cdf(z)
    p_single_hour_interrupt = 1 - p_price_below_bid

    # Clip to valid probability range
    p_single_hour_interrupt = np.clip(p_single_hour_interrupt, 0.001, 0.999)

    p_no_interrupt_over_runtime = (1 - p_single_hour_interrupt) ** runtime_hours
    p_interrupt_over_runtime = 1 - p_no_interrupt_over_runtime

    return p_interrupt_over_runtime
def calculate_utility(bid, predicted_price, residual_std, runtime_hours, budget, risk_profile):
    """
    Scores a candidate bid. Higher utility = better choice.
    """
    p_interrupt = get_interruption_probability(bid, predicted_price, residual_std, runtime_hours)

    expected_cost = bid * runtime_hours

    # Risk profile weighting
    risk_weights = {
        "Conservative": 2.0,   # penalize interruption risk heavily
        "Balanced": 1.0,
        "Aggressive": 0.5      # prioritize cost savings over risk
    }
    risk_weight = risk_weights.get(risk_profile, 1.0)

    # Penalize going over budget heavily
    budget_penalty = 0
    if expected_cost > budget:
        budget_penalty = (expected_cost - budget) * 10

    # Utility: reward low cost + low interruption risk, penalize budget overrun
    utility = -(expected_cost) - (p_interrupt * 100 * risk_weight) - budget_penalty

    return utility, p_interrupt, expected_cost
def find_best_bid(instance_type, availability_zone, operating_system,
                   hour, day_of_week, runtime_hours, budget, risk_profile,
                   min_bid=0.01, max_bid=2.0, step=0.01):
    """
    Decision node: tries candidate bids.
    Chance node: interruption probability per bid (handled inside calculate_utility).
    Picks bid with highest expected utility.
    """
    predicted_price, res_std = get_price_prediction(
        hour, day_of_week, instance_type, availability_zone, operating_system
    )

    best_bid = None
    best_utility = float("-inf")
    best_details = {}

    candidate_bids = np.arange(min_bid, max_bid, step)

    for bid in candidate_bids:
        utility, p_interrupt, expected_cost = calculate_utility(
            bid, predicted_price, res_std, runtime_hours, budget, risk_profile
        )
        if utility > best_utility:
            best_utility = utility
            best_bid = bid
            best_details = {
                "predicted_price": predicted_price,
                "interruption_probability": p_interrupt,
                "expected_cost": expected_cost,
                "completion_probability": 1 - p_interrupt
            }

    return best_bid, best_utility, best_details
def compare_strategies(instance_type, availability_zone, operating_system,
                        hour, day_of_week, runtime_hours, budget, risk_profile):
    """
    Compares the AI-recommended bid against fixed baseline bidding strategies.
    Returns a list of dicts, one per strategy.
    """
    predicted_price, res_std = get_price_prediction(
        hour, day_of_week, instance_type, availability_zone, operating_system
    )

    recommendation = get_recommendation(
        instance_type, availability_zone, operating_system,
        hour, day_of_week, runtime_hours, budget, risk_profile
    )

    strategies = []

    # AI Recommended
    strategies.append({
        "Strategy": "AI Recommended",
        "Bid ($/hr)": recommendation["recommended_bid"],
        "Expected Cost ($)": recommendation["expected_cost"],
        "Interruption Risk (%)": recommendation["interruption_probability"],
        "Completion Probability (%)": recommendation["completion_probability"]
    })

    # Fixed baseline strategies: match predicted price, 20% above, 20% below
    baseline_bids = {
        "Match Market Price": predicted_price,
        "Conservative Fixed (+20%)": predicted_price * 1.2,
        "Aggressive Fixed (-20%)": predicted_price * 0.8,
    }

    for name, bid in baseline_bids.items():
        p_interrupt = get_interruption_probability(bid, predicted_price, res_std, runtime_hours)
        expected_cost = bid * runtime_hours
        strategies.append({
            "Strategy": name,
            "Bid ($/hr)": round(bid, 4),
            "Expected Cost ($)": round(expected_cost, 2),
            "Interruption Risk (%)": round(p_interrupt * 100, 2),
            "Completion Probability (%)": round((1 - p_interrupt) * 100, 2)
        })

    return strategies, predicted_price
if __name__ == "__main__":
    test_cases = [
        {"instance_type": "m4.xlarge", "availability_zone": "us-east-1a",
         "operating_system": "Linux/UNIX", "hour": 9, "day_of_week": 0,
         "runtime_hours": 2, "budget": 3.0, "risk_profile": "Conservative"},

        {"instance_type": "m4.xlarge", "availability_zone": "us-east-1a",
         "operating_system": "Linux/UNIX", "hour": 9, "day_of_week": 0,
         "runtime_hours": 2, "budget": 3.0, "risk_profile": "Aggressive"},

        {"instance_type": "c3.4xlarge", "availability_zone": "us-east-1d",
         "operating_system": "Linux/UNIX", "hour": 22, "day_of_week": 5,
         "runtime_hours": 6, "budget": 10.0, "risk_profile": "Balanced"},
    ]

    for i, case in enumerate(test_cases):
        bid, utility, details = find_best_bid(**case)
        print(f"\n--- Test Case {i+1} ({case['risk_profile']}) ---")
        print(f"Recommended Bid: ${bid:.4f}/hr")
        print(f"Predicted Price: ${details['predicted_price']:.4f}")
        print(f"Interruption Probability: {details['interruption_probability']*100:.2f}%")
        print(f"Expected Cost: ${details['expected_cost']:.2f}")
def get_recommendation(instance_type, availability_zone, operating_system,
                        hour, day_of_week, runtime_hours, budget, risk_profile):
    """
    Main entry point for the dashboard to call.
    Returns a clean dictionary of results.
    """
    bid, utility, details = find_best_bid(
        instance_type=instance_type,
        availability_zone=availability_zone,
        operating_system=operating_system,
        hour=hour,
        day_of_week=day_of_week,
        runtime_hours=runtime_hours,
        budget=budget,
        risk_profile=risk_profile
    )

    on_demand_estimate = details["predicted_price"] * 3  # rough placeholder multiplier
    savings = (on_demand_estimate - details["expected_cost"])
    savings_pct = (savings / on_demand_estimate) * 100 if on_demand_estimate > 0 else 0

    risk_level = "Low"
    if details["interruption_probability"] > 0.4:
        risk_level = "High"
    elif details["interruption_probability"] > 0.15:
        risk_level = "Medium"

    return {
        "recommended_bid": round(bid, 4),
        "expected_cost": round(details["expected_cost"], 2),
        "completion_probability": round(details["completion_probability"] * 100, 2),
        "interruption_probability": round(details["interruption_probability"] * 100, 2),
        "estimated_savings": round(savings, 2),
        "estimated_savings_pct": round(savings_pct, 2),
        "risk_level": risk_level,
        "predicted_price": round(details["predicted_price"], 4)
    }
    print("\n--- Wrapper Function Test ---")
    result = get_recommendation(
        instance_type="m4.xlarge", availability_zone="us-east-1a",
        operating_system="Linux/UNIX", hour=14, day_of_week=2,
        runtime_hours=4, budget=5.0, risk_profile="Balanced"
    )
    print(result)