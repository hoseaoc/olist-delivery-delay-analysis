import pandas as pd

from preprocessing import FEATURES


POLICIES = {
    "baseline": ["baseline_score"],
    "approval": ["risk_score"],
    "carrier": ["urgent", "risk_score"],
}


def add_scores(df, bundle):
    df = df.copy()
    df["baseline_score"] = df["state"].map(bundle["state_rate"]).fillna(bundle["overall_rate"])
    df["risk_score"] = bundle["default_model"].predict_proba(df[FEATURES])[:, 1]
    return df


def daily_snapshot(df, day):
    cutoff = day + pd.Timedelta(days=1)
    active = df.loc[
        df["approved_at"].lt(cutoff)
        & (df["delivered_at"].isna() | df["delivered_at"].ge(cutoff))
    ].copy()
    active["days_left"] = (active["estimate_day"] - day).dt.days
    active["carrier_known"] = active["carrier_at"].lt(cutoff)
    active["urgent"] = active["days_left"].between(1, 2) & ~active["carrier_known"]

    pending = active.loc[active["days_left"].gt(0)]
    overdue = active.loc[active["days_left"].le(0)].sort_values(["estimate_day", "order_id"])
    return pending, overdue


def rank_orders(df, columns):
    return df.sort_values(
        columns + ["approved_at", "order_id"],
        ascending=[False] * len(columns) + [True, True],
    )


def select_queue(ranked, reviewed, capacity):
    queue = ranked.loc[~ranked["order_id"].isin(reviewed)].head(capacity)
    reviewed.update(queue["order_id"])
    return queue


def score_day(df, days, policy, capacity):
    reviewed = set()
    for day in days:
        pending, overdue = daily_snapshot(df, day)
        queue = select_queue(rank_orders(pending, POLICIES[policy]), reviewed, capacity)

    columns = ["order_id", "state", "estimate_day", "days_left", "carrier_known"]
    return queue[columns + POLICIES[policy]], overdue[columns]
