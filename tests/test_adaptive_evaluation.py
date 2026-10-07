import pandas as pd

from src.ingestion.loader import load_log_file
from src.parsing.parser import parse_log_line
from src.preprocessing.dataframe_builder import build_dataframe
from src.features.feature_engineering import create_features
from src.detection.anomaly_detector import detect_anomalies
from src.detection.threshold import (
    calculate_adaptive_threshold,
    apply_threshold
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
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

    # -------------------------
    # 4. Adaptive threshold
    # -------------------------

    scores = predictions["anomaly_score"]

    threshold = calculate_adaptive_threshold(
        scores,
        k=3.0
    )

    adaptive_flags = apply_threshold(
        scores,
        threshold
    )

    # -------------------------
    # 5. Ground truth
    # -------------------------

    truth = pd.read_csv(GROUND_TRUTH_FILE)

    truth["timestamp"] = pd.to_datetime(
        truth["timestamp"]
    )

    results = predictions.merge(
        truth,
        on="timestamp",
        how="left"
    )

    # -------------------------
    # 6. Actual labels
    # -------------------------

    y_true = (
        results["incident_type"] != "normal"
    ).astype(int)

    # -------------------------
    # 7. Adaptive predictions
    # -------------------------

    y_pred = adaptive_flags.astype(int)

    # -------------------------
    # 8. Evaluation
    # -------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

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

    # -------------------------
    # 9. Print results
    # -------------------------

    print("\nAdaptive threshold:", threshold)

    print("\nPrediction counts:")

    print(
        pd.Series(y_pred).value_counts()
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Normal",
                "Incident"
            ],
            zero_division=0
        )
    )

    print("\nMetrics:")

    print({
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })


if __name__ == "__main__":
    main()