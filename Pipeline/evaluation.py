import pandas as pd
from sklearn.metrics import average_precision_score

from preprocessing import FEATURES
from scoring import POLICIES, add_scores, daily_snapshot, rank_orders, select_queue


def evaluate_approval(test, days, bundle):
    test = add_scores(test, bundle)
    test["tuned_score"] = bundle["tuned_model"].predict_proba(test[FEATURES])[:, 1]
    results = []
    for method, score in [
        ("baseline", "baseline_score"),
        ("default", "risk_score"),
        ("tuned", "tuned_score"),
    ]:
        queue = rank_orders(test, [score]).groupby("approved_day").head(20)
        found = int(queue["late"].sum())
        results.append({
            "method": method,
            "reviewed": len(queue),
            "late_found": found,
            "late_missed": int(test["late"].sum()) - found,
            "late_found_per_day": found / len(days),
            "recall": found / test["late"].sum(),
            "precision": queue["late"].mean(),
            "average_precision": average_precision_score(test["late"], test[score]),
        })
    return pd.DataFrame(results)


def evaluate_replay(cohort, days, bundle):
    scored = add_scores(cohort, bundle)
    pending_by_day = [daily_snapshot(scored, day)[0] for day in days]
    eligible = pd.concat(pending_by_day).drop_duplicates("order_id")
    eligible_late = int(eligible["late"].sum())
    results = []

    for policy, columns in POLICIES.items():
        ranked_days = [rank_orders(pending, columns) for pending in pending_by_day]
        for capacity in [20, 50, 100]:
            reviewed = set()
            selected = pd.concat([
                select_queue(ranked, reviewed, capacity) for ranked in ranked_days
            ])
            found = int(selected["late"].sum())
            results.append({
                "policy": policy,
                "daily_capacity": capacity,
                "reviewed": len(selected),
                "eligible_late": eligible_late,
                "late_found": found,
                "late_missed": eligible_late - found,
                "recall": found / eligible_late,
                "median_days_before_deadline": selected.loc[selected["late"], "days_left"].median(),
                "canceled_or_unavailable": int(selected["canceled_or_unavailable"].sum()),
                "unknown_outcome": int((~selected["has_outcome"] & ~selected["canceled_or_unavailable"]).sum()),
            })
    return pd.DataFrame(results)
