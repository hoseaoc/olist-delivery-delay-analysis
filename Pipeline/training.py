import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import FEATURES


def fit_model(df, c=1.0):
    preprocess = ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore"), ["state", "weekday"]),
        ("number", StandardScaler(), ["days_to_estimate"]),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(C=c, max_iter=1000)),
    ])
    return model.fit(df[FEATURES], df["late"])


def train_models(train_tune, validation, train):
    results = []
    for c in [1.0, 0.001, 0.01, 0.1, 10.0]:
        model = fit_model(train_tune, c)
        scores = model.predict_proba(validation[FEATURES])[:, 1]
        results.append({
            "C": c,
            "validation_average_precision": average_precision_score(validation["late"], scores),
        })

    tuning = pd.DataFrame(results).sort_values(
        "validation_average_precision", ascending=False, kind="stable"
    )
    bundle = {
        "default_model": fit_model(train),
        "tuned_model": fit_model(train, c=float(tuning.iloc[0]["C"])),
        "state_rate": train.groupby("state")["late"].mean(),
        "overall_rate": train["late"].mean(),
    }
    return bundle, tuning
