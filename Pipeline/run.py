import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import pandas as pd

from evaluation import evaluate_approval, evaluate_replay
from ingestion import ingest
from preprocessing import TEST_END, TEST_START, preprocess, split_data, test_cohort
from scoring import POLICIES, add_scores, score_day
from training import train_models


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "Pipeline" / "output"
MODEL_PATH = OUTPUT / "models.joblib"


def write_artifact(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        write(temp_path)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def save_csv(df, path):
    write_artifact(path, lambda temp: df.to_csv(temp, index=False))


def train(df):
    train_tune, validation, train_final, test = split_data(df)
    bundle, tuning = train_models(train_tune, validation, train_final)
    days = pd.date_range(TEST_START, TEST_END, inclusive="left")
    approval = evaluate_approval(test, days, bundle)
    replay = evaluate_replay(test_cohort(df), days, bundle)

    save_csv(tuning, OUTPUT / "tuning.csv")
    save_csv(approval, OUTPUT / "approval_metrics.csv")
    save_csv(replay, OUTPUT / "replay_metrics.csv")
    write_artifact(MODEL_PATH, lambda temp: joblib.dump(bundle, temp))
    print(f"Train: {len(train_final):,} | Test: {len(test):,}")
    print(replay[["policy", "daily_capacity", "late_found", "recall"]].to_string(index=False))
    print(f"Output training dan evaluasi: {OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline prediksi risiko pengiriman Olist")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="Latih, evaluasi, dan simpan model")
    score = commands.add_parser("score", help="Hasilkan antrean historis untuk satu tanggal")
    score.add_argument("--date", required=True, help="Tanggal replay, YYYY-MM-DD")
    score.add_argument("--capacity", type=int, choices=[20, 50, 100], default=20)
    score.add_argument("--policy", choices=POLICIES, default="carrier")
    args = parser.parse_args()

    if args.command == "train":
        train(preprocess(ingest(ROOT / "Data")))
        return

    try:
        day = pd.Timestamp(args.date).normalize()
    except ValueError:
        parser.error("Gunakan tanggal dengan format YYYY-MM-DD.")
    if pd.isna(day) or day.tzinfo is not None or not TEST_START <= day < TEST_END:
        parser.error("Tanggal replay harus berada antara 2018-06-01 dan 2018-08-31.")
    if not MODEL_PATH.is_file():
        parser.error("Jalankan perintah train untuk membuat model terlebih dahulu.")

    df = preprocess(ingest(ROOT / "Data"))
    scored = add_scores(test_cohort(df), joblib.load(MODEL_PATH))
    queue, overdue = score_day(scored, pd.date_range(TEST_START, day), args.policy, args.capacity)

    day_output = OUTPUT / f"{day:%Y-%m-%d}_{args.policy}_{args.capacity}"
    queue = queue.reset_index(drop=True)
    queue.insert(0, "priority", queue.index + 1)
    save_csv(queue, day_output / "risk_queue.csv")
    save_csv(overdue, day_output / "overdue.csv")
    print(f"Antrean risiko: {len(queue)} | Lewat estimasi: {len(overdue)}")
    print(f"Output scoring: {day_output}")


if __name__ == "__main__":
    main()
