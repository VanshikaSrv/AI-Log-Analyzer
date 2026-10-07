import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from src.ingestion.loader import load_log_file
from src.parsing.parser import parse_log_line
from src.preprocessing.dataframe_builder import build_dataframe
from src.features.feature_engineering import create_features
from src.detection.anomaly_detector import detect_anomalies
from src.detection.threshold import (
    calculate_adaptive_threshold,
    apply_threshold
)


LOG_FILE = "data/raw/generated_application.log"
GROUND_TRUTH_FILE = "data/processed/ground_truth.csv"


def main():

    # -------------------------
    # 1. Load and parse logs
    # -------------------------

    lines = load_log_file(LOG_FILE)

    events = []

    for line in lines:

        event = parse_log_line(line)

        if event is not None:
            events.append(event)

    # -------------------------
    # 2. Build features
    # -------------------------

    df = build_dataframe(events)

    features = create_features(df)

    # -------------------------
    # 3. Generate anomaly scores
    # -------------------------

    predictions = detect_anomalies(
        features,
        contamination=0.05
    )

    scores = predictions["anomaly_score"]

    # -------------------------
    # 4. Load ground truth
    # -------------------------

    truth = pd.read_csv(
        GROUND_TRUTH_FILE
    )

    truth["timestamp"] = pd.to_datetime(
        truth["timestamp"]
    )

    results = predictions.merge(
        truth,
        on="timestamp",
        how="left"
    )

    y_true = (
        results["incident_type"] != "normal"
    ).astype(int)

    # -------------------------
    # 5. Test different k values
    # -------------------------

    k_values = [
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0
    ]

    print("\n===== THRESHOLD SENSITIVITY =====\n")

    print(
        f"{'k':<6}"
        f"{'Threshold':<15}"
        f"{'Alerts':<10}"
        f"{'TP':<6}"
        f"{'FP':<6}"
        f"{'FN':<6}"
        f"{'Precision':<12}"
        f"{'Recall':<10}"
        f"{'F1':<10}"
    )

    print("-" * 85)

    for k in k_values:

        # Calculate threshold
        threshold = calculate_adaptive_threshold(
            scores,
            k=k
        )

        # Apply threshold
        flags = apply_threshold(
            scores,
            threshold
        )

        y_pred = flags.astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred
        ).ravel()

        # Metrics
        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        print(
            f"{k:<6.1f}"
            f"{threshold:<15.6f}"
            f"{flags.sum():<10}"
            f"{tp:<6}"
            f"{fp:<6}"
            f"{fn:<6}"
            f"{precision:<12.3f}"
            f"{recall:<10.3f}"
            f"{f1:<10.3f}"
        )


if __name__ == "__main__":
    main()