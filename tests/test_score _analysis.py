import pandas as pd

from src.ingestion.loader import load_log_file
from src.parsing.parser import parse_log_line
from src.preprocessing.dataframe_builder import build_dataframe
from src.features.feature_engineering import create_features
from src.detection.anomaly_detector import detect_anomalies


LOG_FILE = "data/raw/generated_application.log"
GROUND_TRUTH_FILE = "data/processed/ground_truth.csv"


def main():

    # 1. Load and parse logs
    lines = load_log_file(LOG_FILE)

    events = []

    for line in lines:
        event = parse_log_line(line)

        if event is not None:
            events.append(event)

    # 2. Build features
    df = build_dataframe(events)
    features = create_features(df)

    # 3. Generate anomaly scores
    predictions = detect_anomalies(
        features,
        contamination=0.05
    )

    # 4. Load ground truth
    truth = pd.read_csv(GROUND_TRUTH_FILE)
    truth["timestamp"] = pd.to_datetime(truth["timestamp"])

    # 5. Combine scores with ground truth
    results = predictions.merge(
        truth,
        on="timestamp",
        how="left"
    )

    # --------------------------------
    # Overall score distribution
    # --------------------------------

    print("\n===== OVERALL SCORE DISTRIBUTION =====")

    print(
        results["anomaly_score"].describe()
    )

    # --------------------------------
    # Normal score distribution
    # --------------------------------

    normal_scores = results[
        results["incident_type"] == "normal"
    ]["anomaly_score"]

    print("\n===== NORMAL WINDOWS =====")

    print(
        normal_scores.describe()
    )

    # --------------------------------
    # Incident score distribution
    # --------------------------------

    incident_scores = results[
        results["incident_type"] != "normal"
    ]["anomaly_score"]

    print("\n===== INCIDENT WINDOWS =====")

    print(
        incident_scores.describe()
    )

    # --------------------------------
    # Score statistics by incident
    # --------------------------------

    print("\n===== SCORE BY INCIDENT TYPE =====")

    print(
        results.groupby("incident_type")["anomaly_score"]
        .agg(["count", "min", "mean", "median", "max"])
    )

    # --------------------------------
    # Important percentiles
    # --------------------------------

    print("\n===== SCORE PERCENTILES =====")

    percentiles = [1, 5, 10, 15, 20, 25, 50, 75, 90, 95, 99]

    for p in percentiles:

        value = results["anomaly_score"].quantile(
            p / 100
        )

        print(
            f"{p:>2}th percentile: {value:.6f}"
        )


if __name__ == "__main__":
    main()