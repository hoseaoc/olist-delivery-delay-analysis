import pandas as pd


FEATURES = ["state", "weekday", "days_to_estimate"]
VALIDATION_START = pd.Timestamp("2018-04-01")
VALIDATION_END = pd.Timestamp("2018-05-01")
TEST_START = pd.Timestamp("2018-06-01")
TEST_END = pd.Timestamp("2018-09-01")


def preprocess(df):
    df = df.rename(columns={
        "order_approved_at": "approved_at",
        "order_delivered_customer_date": "delivered_at",
        "order_estimated_delivery_date": "estimated_at",
        "order_delivered_carrier_date": "carrier_at",
    })
    for col in ["approved_at", "delivered_at", "estimated_at", "carrier_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["approved_at", "estimated_at"]).copy()
    df["approved_day"] = df["approved_at"].dt.normalize()
    df["estimate_day"] = df["estimated_at"].dt.normalize()
    df["state"] = df["customer_state"].fillna("Tidak diketahui")
    df["weekday"] = df["approved_at"].dt.day_name()
    df["days_to_estimate"] = (df["estimate_day"] - df["approved_day"]).dt.days

    df["has_outcome"] = df["order_status"].eq("delivered") & df["delivered_at"].notna()
    df["late"] = df["has_outcome"] & df["delivered_at"].dt.normalize().gt(df["estimate_day"])
    df["canceled_or_unavailable"] = df["order_status"].isin(["canceled", "unavailable"])
    return df


def split_data(df):
    labeled = df.loc[df["has_outcome"]]
    train = labeled.loc[
        labeled["approved_at"].lt(TEST_START) & labeled["delivered_at"].lt(TEST_START)
    ]
    train_tune = train.loc[
        train["approved_at"].lt(VALIDATION_START) & train["delivered_at"].lt(VALIDATION_START)
    ]
    validation = train.loc[
        train["approved_at"].between(VALIDATION_START, VALIDATION_END, inclusive="left")
    ]
    test = test_cohort(labeled)
    return train_tune, validation, train, test


def test_cohort(df):
    return df.loc[df["approved_at"].between(TEST_START, TEST_END, inclusive="left")].copy()
