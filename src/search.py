import math
import numpy as np

from clean_data import get_price_stats

# Approximate us-east-1 on-demand $/hr. VERIFY on the AWS pricing page
# and add Windows / SUSE rows before presenting.
ON_DEMAND_PRICES = {
    ("m4.xlarge", "Linux/UNIX"): 0.20,
    ("c3.4xlarge", "Linux/UNIX"): 0.84,
    ("r3.large", "Linux/UNIX"): 0.166,
}

RISK_SETTINGS = {
    "Conservative": {"worst_q": 99, "penalty": 6.0},
    "Balanced":     {"worst_q": 97, "penalty": 3.0},
    "Aggressive":   {"worst_q": 90, "penalty": 1.0},
}

# Candidate bids are real price quantiles from the history, so each
# candidate corresponds to a genuine interruption probability.
QUANTILES = [10, 25, 50, 60, 70, 80, 85, 90, 93, 95, 97, 98, 99, 99.5, 100]


# ---------------- alpha-beta game tree ----------------
def _leaf_value(cost, progress, ctx):
    """Heuristic evaluation at the search cutoff."""
    remaining = ctx["runtime"] - progress
    return -(cost + remaining * ctx["typical"] * (1 + ctx["penalty"]))


def _max_node(level, depth, cost, progress, alpha, beta, ctx, stats):
    """MAX = user chooses a bid for this hour."""
    if level == depth:
        stats["nodes"] += 1
        return _leaf_value(cost, progress, ctx), None

    best_val, best_bid = -math.inf, None
    for bid in ctx["bids"]:                      # ascending bids
        stats["nodes"] += 1
        val = _min_node(level, depth, bid, cost, progress,
                        alpha, beta, ctx, stats)
        if val > best_val:                       # strict: ties keep the lower bid
            best_val, best_bid = val, bid
        alpha = max(alpha, best_val)
        if ctx["prune"] and alpha >= beta:
            stats["cutoffs"] += 1
            break
    return best_val, best_bid


def _min_node(level, depth, bid, cost, progress, alpha, beta, ctx, stats):
    """MIN = market picks the worst price scenario for this hour."""
    worst = math.inf
    for price in ctx["scenarios"]:               # worst scenario first
        stats["nodes"] += 1
        if price <= bid:                         # hour completes, pay market price
            new_cost, new_progress = cost + price, progress + 1
        else:                                    # interrupted, hour lost
            new_cost, new_progress = cost, progress
        val, _ = _max_node(level + 1, depth, new_cost, new_progress,
                           alpha, beta, ctx, stats)
        worst = min(worst, val)
        beta = min(beta, worst)
        if ctx["prune"] and beta <= alpha:
            stats["cutoffs"] += 1
            break
    return worst


# ---------------- helpers (statistics only, no ML) ----------------
def _exceedance(prices, bid):
    return float((prices > bid).mean())


def _interrupt_over_runtime(prices, bid, runtime_hours):
    p = _exceedance(prices, bid)
    return 1 - (1 - p) ** runtime_hours


def _infeasible(reason):
    return {"infeasible": True, "reason": reason}


# ---------------- main entry point for the dashboard ----------------
def get_recommendation(instance_type, availability_zone, operating_system,
                       hour, day_of_week, runtime_hours, budget,
                       risk_profile, deadline_hours=None):
    prices = get_price_stats(instance_type, availability_zone,
                             operating_system, hour)
    if len(prices) == 0:
        return _infeasible("No historical data for this instance / zone / OS combination.")

    # CSP constraint C3: deadline
    if deadline_hours is not None and deadline_hours < runtime_hours:
        return _infeasible(
            f"Deadline ({deadline_hours}h) is shorter than the runtime ({runtime_hours}h)."
        )

    settings = RISK_SETTINGS.get(risk_profile, RISK_SETTINGS["Balanced"])
    low = float(np.percentile(prices, 25))
    typical = float(np.percentile(prices, 50))
    worst = float(np.percentile(prices, settings["worst_q"]))

    on_demand = ON_DEMAND_PRICES.get((instance_type, operating_system))
    cap = on_demand if on_demand else float(prices.max())

    # Candidate bids (domain of the CSP variable) = real price quantiles
    candidates = np.unique(np.percentile(prices, QUANTILES))

    # CSP domain filtering: C1 (budget) and C2 (price cap)
    feasible = [float(b) for b in candidates
                if b * runtime_hours <= budget and b <= cap]
    if not feasible:
        return _infeasible(
            f"No bid fits: budget ${budget} cannot cover {runtime_hours}h even at the "
            f"lowest candidate bid (${candidates[0]:.3f}/hr). Increase the budget or "
            f"reduce the runtime."
        )

    # The market cannot be worse than what the constraints let us bid against
    worst = min(worst, max(feasible))

    ctx = {
        "bids": feasible,
        "scenarios": [worst, typical, low],      # worst first (better pruning)
        "typical": typical,
        "penalty": settings["penalty"],
        "runtime": runtime_hours,
        "prune": True,
    }
    depth = max(1, min(int(runtime_hours), 3))

    # Alpha-beta search
    stats = {"nodes": 0, "cutoffs": 0}
    _, bid = _max_node(0, depth, 0.0, 0, -math.inf, math.inf, ctx, stats)

    # Same search WITHOUT pruning (plain minimax) for comparison
    stats_full = {"nodes": 0, "cutoffs": 0}
    _max_node(0, depth, 0.0, 0, -math.inf, math.inf, dict(ctx, prune=False), stats_full)

    # Reported figures from historical frequencies
    p_interrupt = _interrupt_over_runtime(prices, bid, runtime_hours)
    expected_spend = min(typical, bid) * runtime_hours
    max_spend = bid * runtime_hours

    on_demand_hourly = on_demand if on_demand else float(np.percentile(prices, 99))
    on_demand_total = on_demand_hourly * runtime_hours
    savings = on_demand_total - expected_spend
    savings_pct = (savings / on_demand_total) * 100 if on_demand_total > 0 else 0

    risk_level = "Low"
    if p_interrupt > 0.4:
        risk_level = "High"
    elif p_interrupt > 0.15:
        risk_level = "Medium"

    return {
        "recommended_bid": round(bid, 4),
        "expected_cost": round(expected_spend, 2),
        "max_cost": round(max_spend, 2),
        "completion_probability": round((1 - p_interrupt) * 100, 2),
        "interruption_probability": round(p_interrupt * 100, 2),
        "estimated_savings": round(savings, 2),
        "estimated_savings_pct": round(savings_pct, 2),
        "risk_level": risk_level,
        "predicted_price": round(typical, 4),
        "nodes_visited": stats["nodes"],
        "nodes_without_pruning": stats_full["nodes"],
        "pruning_cutoffs": stats["cutoffs"],
    }

def compare_strategies(instance_type, availability_zone, operating_system,
                       hour, day_of_week, runtime_hours, budget, risk_profile):
    """AI recommendation vs fixed baseline bids. Returns (rows, median_price)."""
    prices = get_price_stats(instance_type, availability_zone, operating_system, hour)
    rec = get_recommendation(instance_type, availability_zone, operating_system,
                             hour, day_of_week, runtime_hours, budget, risk_profile)
    if rec.get("infeasible"):
        return [], 0.0

    typical = rec["predicted_price"]
    rows = [{
        "Strategy": "AI Recommended (Alpha-Beta)",
        "Bid ($/hr)": rec["recommended_bid"],
        "Expected Cost ($)": rec["expected_cost"],
        "Interruption Risk (%)": rec["interruption_probability"],
        "Completion Probability (%)": rec["completion_probability"],
    }]

    baselines = {
        "Match Median Price": float(np.percentile(prices, 50)),
        "Conservative Fixed (P90)": float(np.percentile(prices, 90)),
        "Aggressive Fixed (P25)": float(np.percentile(prices, 25)),
    }
    for name, bid in baselines.items():
        p_int = _interrupt_over_runtime(prices, bid, runtime_hours)
        rows.append({
            "Strategy": name,
            "Bid ($/hr)": round(bid, 4),
            "Expected Cost ($)": round(min(typical, bid) * runtime_hours, 2),
            "Interruption Risk (%)": round(p_int * 100, 2),
            "Completion Probability (%)": round((1 - p_int) * 100, 2),
        })
    return rows, typical


if __name__ == "__main__":
    for profile in ["Conservative", "Balanced", "Aggressive"]:
        print(profile, get_recommendation(
            "m4.xlarge", "us-east-1a", "Linux/UNIX",
            hour=14, day_of_week=2, runtime_hours=4,
            budget=11.0, risk_profile=profile, deadline_hours=6))