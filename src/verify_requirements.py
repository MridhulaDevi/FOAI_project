"""
Automated check of the project requirements:
  1. No ML / no training anywhere in the code
  2. Alpha-beta pruning is correct (same answer as plain minimax) and prunes
  3. CSP constraints work (budget / deadline)
  4. Risk profiles behave sensibly
  5. Forward chaining really chains
Run from the project root:  python src/verify_requirements.py
"""
import os
import sys
import math
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import search
from explain import get_explanation, forward_chain, build_facts

results = []


def check(name, passed, detail=""):
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))


print("\n=== 1. NO ML / NO TRAINING ===")
banned = ["sklearn", "scikit", "joblib", "scipy", "tensorflow", "torch", "keras",
          "xgboost", "LinearRegression", "RandomForest", "train_test_split",
          ".fit(", ".predict(", "pickle"]
src_dir = os.path.join(BASE_DIR, "src")
found = []
for fname in os.listdir(src_dir):
    if fname.endswith(".py") and fname != "verify_requirements.py":
        with open(os.path.join(src_dir, fname), encoding="utf-8") as f:
            text = f.read()
        for token in banned:
            if token in text:
                found.append(f"{fname}: {token}")
check("No ML libraries / training calls in src/*.py", not found, ", ".join(found))

model_files = []
for root, _, files in os.walk(BASE_DIR):
    if "venv" in root or ".git" in root:
        continue
    model_files += [os.path.join(root, x) for x in files if x.endswith((".pkl", ".joblib", ".h5"))]
check("No saved model files (.pkl / .joblib / .h5)", not model_files, ", ".join(model_files))

req_path = os.path.join(BASE_DIR, "requirements.txt")
if os.path.exists(req_path):
    with open(req_path, encoding="utf-8", errors="ignore") as f:
        req = f.read().lower()
    bad = [x for x in ["scikit-learn", "joblib", "scipy"] if x in req]
    check("requirements.txt has no ML packages", not bad, ", ".join(bad))
check("train_model.py deleted", not os.path.exists(os.path.join(src_dir, "train_model.py")))


print("\n=== 2. ALPHA-BETA CORRECTNESS ===")
random.seed(42)
mismatch, total_pruned, total_full, runs = 0, 0, 0, 300
for _ in range(runs):
    n_bids = random.randint(3, 15)
    bids = sorted(random.uniform(0.02, 0.6) for _ in range(n_bids))
    scen = sorted((random.uniform(0.02, 0.6) for _ in range(3)), reverse=True)
    ctx = {"bids": bids, "scenarios": scen, "typical": scen[1],
           "penalty": random.choice([1.0, 3.0, 6.0]), "runtime": random.randint(1, 8), "prune": True}
    depth = max(1, min(ctx["runtime"], 3))
    s1, s2 = {"nodes": 0, "cutoffs": 0}, {"nodes": 0, "cutoffs": 0}
    v1, b1 = search._max_node(0, depth, 0.0, 0, -math.inf, math.inf, ctx, s1)
    v2, b2 = search._max_node(0, depth, 0.0, 0, -math.inf, math.inf, dict(ctx, prune=False), s2)
    if abs(v1 - v2) > 1e-9 or b1 != b2:
        mismatch += 1
    total_pruned += s1["nodes"]
    total_full += s2["nodes"]
check(f"Alpha-beta gives the SAME value and bid as plain minimax ({runs} random trees)",
      mismatch == 0, f"{mismatch} mismatches")
check("Pruning visits fewer nodes than plain minimax", total_pruned < total_full,
      f"{total_pruned} vs {total_full} nodes ({100 * (1 - total_pruned / total_full):.1f}% fewer)")


print("\n=== 3. FULL PIPELINE + CSP CONSTRAINTS ===")
args = dict(instance_type="m4.xlarge", availability_zone="us-east-1a",
            operating_system="Linux/UNIX", hour=14, day_of_week=2)
recs = {}
for profile in ["Conservative", "Balanced", "Aggressive"]:
    recs[profile] = search.get_recommendation(**args, runtime_hours=4, budget=11.0,
                                              risk_profile=profile, deadline_hours=6)
    r = recs[profile]
    check(f"{profile}: recommendation returned with all fields",
          all(k in r for k in ["recommended_bid", "completion_probability", "nodes_visited"]),
          f"bid={r.get('recommended_bid')}")
    check(f"{profile}: pruning cutoffs happened",
          r["pruning_cutoffs"] > 0 and r["nodes_visited"] < r["nodes_without_pruning"],
          f"{r['nodes_visited']} vs {r['nodes_without_pruning']} nodes")

for budget in [0.3, 1.0, 5.0, 11.0]:
    r = search.get_recommendation(**args, runtime_hours=4, budget=budget,
                                  risk_profile="Balanced", deadline_hours=6)
    ok = r.get("infeasible") or r["max_cost"] <= budget + 1e-9
    check(f"Budget constraint respected (budget ${budget})", ok,
          "infeasible message" if r.get("infeasible") else f"max cost ${r['max_cost']}")

r = search.get_recommendation(**args, runtime_hours=4, budget=0.05, risk_profile="Balanced", deadline_hours=6)
check("Tiny budget -> 'no feasible bid' message", r.get("infeasible") is True)
r = search.get_recommendation(**args, runtime_hours=4, budget=11.0, risk_profile="Balanced", deadline_hours=2)
check("Deadline shorter than runtime -> infeasible message", r.get("infeasible") is True)

rows, _ = search.compare_strategies(**args, runtime_hours=4, budget=11.0, risk_profile="Balanced")
check("Strategy comparison returns AI + 3 baselines", len(rows) == 4)


print("\n=== 4. RISK PROFILE BEHAVIOUR ===")
c, b, a = recs["Conservative"], recs["Balanced"], recs["Aggressive"]
check("Bid order: Conservative >= Balanced >= Aggressive",
      c["recommended_bid"] >= b["recommended_bid"] >= a["recommended_bid"],
      f"{c['recommended_bid']} / {b['recommended_bid']} / {a['recommended_bid']}")
check("Completion order: Conservative >= Balanced >= Aggressive",
      c["completion_probability"] >= b["completion_probability"] >= a["completion_probability"],
      f"{c['completion_probability']} / {b['completion_probability']} / {a['completion_probability']}")
check("Profiles are NOT all identical",
      len({c["recommended_bid"], b["recommended_bid"], a["recommended_bid"]}) > 1)


print("\n=== 5. FORWARD CHAINING ===")
fake_rec = {"interruption_probability": 55.0, "completion_probability": 45.0, "expected_cost": 4.0,
            "risk_level": "High", "estimated_savings_pct": 50.0}
facts = build_facts(fake_rec, budget=10.0, runtime_hours=8, deadline_hours=9, risk_profile="Aggressive")
_, out = forward_chain(facts)
fired = out["rules_fired"]
check("Base rules fire first (R1, R3)", "R1" in fired and "R3" in fired, str(fired))
check("Chained rule R7 fires from DERIVED facts", "R7" in fired and fired.index("R7") > fired.index("R1"))
check("Second-level chain R9 fires from R7's conclusion", "R9" in fired)
check("Derived facts were added (risk_high, time_pressure, advise_raise_bid)",
      all(out.get(k) for k in ["risk_high", "time_pressure", "advise_raise_bid"]))
text, _ = get_explanation(recs["Balanced"], 11.0, 4, 6, "Balanced")
check("Explanation text produced", len(text) > 20)

print("\n" + "=" * 50)
print(f"RESULT: {sum(results)}/{len(results)} checks passed")
print("ALL REQUIREMENTS VERIFIED" if all(results) else "SOME CHECKS FAILED - see [FAIL] lines above")