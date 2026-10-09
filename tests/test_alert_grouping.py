import pandas as pd

from src.pipeline.log_pipeline import process_log_file
from src.features.feature_engineering import create_features

from src.detection.anomaly_detector import (
    fit_detector,
    score_data,
)

from src.detection.threshold import (
    calculate_adaptive_threshold,
    apply_threshold,
)

from src.incidents.alert_grouping import group_alerts


LOG_FILE = "data/raw/generated_application_test2.log"
GROUND_TRUTH_FILE = "data/processed/ground_truth_test2.csv"


def main():

    # ==========================================
    # 1. Process logs
    # ==========================================

    df = process_log_file(LOG_FILE)

    features = create_features(df)

    features = (
        features
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


    # ==========================================
    # 2. Time-aware split
    # ==========================================

    split_index = int(len(features) * 0.70)

    baseline_features = features.iloc[:split_index].copy()
    test_features = features.iloc[split_index:].copy()


    # ==========================================
    # 3. Train detector
    # ==========================================

    model = fit_detector(
        baseline_features,
        contamination=0.05
    )


    # ==========================================
    # 4. Score baseline + test
    # ==========================================

    baseline = score_data(
        model,
        baseline_features
    )

    test = score_data(
        model,
        test_features
    )


    # ==========================================
    # 5. Adaptive threshold
    # ==========================================

    threshold = calculate_adaptive_threshold(
        baseline["anomaly_score"],
        k=3.0
    )

    test["adaptive_anomaly"] = apply_threshold(
        test["anomaly_score"],
        threshold
    )


    # ==========================================
    # 6. Group alerts
    # ==========================================

    incidents = group_alerts(test)


    # ==========================================
    # 7. Load ground truth
    # ==========================================

    truth = pd.read_csv(GROUND_TRUTH_FILE)

    truth["timestamp"] = pd.to_datetime(
        truth["timestamp"]
    )


    # ==========================================
    # 8. Match test windows with ground truth
    # ==========================================

    test_with_truth = test.merge(
        truth,
        on="timestamp",
        how="left"
    )


    # ==========================================
    # 9. Show actual incident windows
    # ==========================================

    actual_incidents = test_with_truth[
        test_with_truth["incident_type"] != "normal"
    ].copy()


    print("\n========== DETECTED INCIDENTS ==========")

    if incidents.empty:
        print("No incidents detected.")
    else:
        print(
            incidents.to_string(index=False)
        )


    print("\n========== ACTUAL INCIDENT WINDOWS ==========")

    if actual_incidents.empty:
        print("No actual incidents in test period.")
    else:
        print(
            actual_incidents[
                ["timestamp", "incident_type"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()