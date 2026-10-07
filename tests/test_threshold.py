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


LOG_FILE = "data/raw/generated_application.log"


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
    # 3. Get Isolation Forest scores
    # -------------------------

    predictions = detect_anomalies(
        features,
        contamination=0.05
    )

    scores = predictions["anomaly_score"]

    # -------------------------
    # 4. Calculate adaptive threshold
    # -------------------------

    threshold = calculate_adaptive_threshold(
        scores,
        k=3.0
    )

    # -------------------------
    # 5. Apply threshold
    # -------------------------

    anomaly_flags = apply_threshold(
        scores,
        threshold
    )

    # -------------------------
    # 6. Display results
    # -------------------------

    print("\nAdaptive threshold:", threshold)

    print("\nNumber of windows:", len(scores))

    print(
        "Number of anomalies:",
        anomaly_flags.sum()
    )

    print(
        "Number of normal windows:",
        (~anomaly_flags).sum()
    )

    print("\nScore statistics:")

    print(scores.describe())

    print("\nLowest scores:")

    print(
        scores.sort_values().head(10)
    )


if __name__ == "__main__":
    main()