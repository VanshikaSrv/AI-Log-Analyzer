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


def main():

    # ==========================================
    # 1. Load and process logs
    # ==========================================

    df = process_log_file(LOG_FILE)

    print("\n========== DATASET ==========")
    print(f"Total logs: {len(df)}")
    print(f"First timestamp: {df['timestamp'].iloc[0]}")
    print(f"Last timestamp: {df['timestamp'].iloc[-1]}")


    # ==========================================
    # 2. Create 5-minute features
    # ==========================================

    features = create_features(df)

    features = (
        features
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print("\n========== FEATURES ==========")
    print(f"Total feature windows: {len(features)}")


    # ==========================================
    # 3. Time-aware split
    # ==========================================

    split_index = int(len(features) * 0.70)

    baseline_features = features.iloc[:split_index].copy()
    test_features = features.iloc[split_index:].copy()

    print("\n========== TIME SPLIT ==========")
    print(f"Baseline windows: {len(baseline_features)}")
    print(f"Test windows: {len(test_features)}")


    # ==========================================
    # 4. Train detector on baseline
    # ==========================================

    model = fit_detector(
        baseline_features,
        contamination=0.05
    )


    # ==========================================
    # 5. Score baseline and test data
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
    # 6. Calculate adaptive threshold
    # ==========================================

    threshold = calculate_adaptive_threshold(
        baseline["anomaly_score"],
        k=3.0
    )

    print("\n========== THRESHOLD ==========")
    print(f"Adaptive threshold: {threshold:.6f}")


    # ==========================================
    # 7. Convert scores into anomaly alerts
    # ==========================================

    test["adaptive_anomaly"] = apply_threshold(
        test["anomaly_score"],
        threshold
    )

    print("\n========== ALERTS ==========")
    print(
        f"Total test windows: {len(test)}"
    )

    print(
        f"Alerts: {test['adaptive_anomaly'].sum()}"
    )

    print(
        f"Normal: {(~test['adaptive_anomaly']).sum()}"
    )


    # ==========================================
    # 8. Group alerts into incidents
    # ==========================================

    incidents = group_alerts(test)


    # ==========================================
    # 9. Display incidents
    # ==========================================

    print("\n========== GROUPED INCIDENTS ==========")

    if incidents.empty:

        print("No incidents detected.")

    else:

        print(incidents.to_string(index=False))


if __name__ == "__main__":
    main()