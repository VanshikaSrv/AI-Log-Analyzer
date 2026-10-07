import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.pipeline.log_pipeline import process_log_file
from src.features.feature_engineering import create_features
from src.detection.anomaly_detector import fit_detector, score_data
from src.detection.threshold import (
    calculate_adaptive_threshold,
    apply_threshold,
)


# Test 2 dataset
LOG_FILE = "data/raw/generated_application_test2.log"
GROUND_TRUTH_FILE = "data/processed/ground_truth_test2.csv"


def main():

    # ============================================================
    # 1. Process raw logs
    # ============================================================

    df = process_log_file(LOG_FILE)

    print("\n========== DATASET CHECK ==========")
    print(f"Log file: {LOG_FILE}")
    print(f"Total parsed logs: {len(df)}")
    print(f"First timestamp: {df['timestamp'].iloc[0]}")
    print(f"Last timestamp: {df['timestamp'].iloc[-1]}")

    # ============================================================
    # 2. Create time-window features
    # ============================================================

    features = create_features(df)

    features = features.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # ============================================================
    # 3. Chronological 70/30 split
    #
    # First 70% = historical baseline
    # Last 30%  = future/test data
    # ============================================================

    split_index = int(len(features) * 0.70)

    baseline_features = features.iloc[:split_index].copy()
    test_features = features.iloc[split_index:].copy()

    # ============================================================
    # 4. Train Isolation Forest ONLY on historical data
    # ============================================================

    model = fit_detector(
        baseline_features,
        contamination=0.05,
    )

    # ============================================================
    # 5. Score historical data
    # ============================================================

    baseline = score_data(
        model,
        baseline_features,
    )

    # ============================================================
    # 6. Score future data using historical model
    # ============================================================

    test = score_data(
        model,
        test_features,
    )

    # ============================================================
    # 7. Calculate adaptive threshold
    #    ONLY from historical baseline scores
    # ============================================================

    threshold_val = calculate_adaptive_threshold(
        baseline["anomaly_score"],
        k=3.0,
    )

    # ============================================================
    # 8. Apply historical threshold to future data
    # ============================================================

    test["adaptive_anomaly"] = apply_threshold(
        test["anomaly_score"],
        threshold_val,
    )

    # ============================================================
    # 9. Display test results
    # ============================================================

    print("\n========== TEST RESULTS ==========")

    print(f"Test windows: {len(test)}")

    print(
        f"Alerts raised: "
        f"{test['adaptive_anomaly'].sum()}"
    )

    print(
        f"Normal windows: "
        f"{(~test['adaptive_anomaly']).sum()}"
    )

    # ============================================================
    # 10. Display adaptive threshold
    # ============================================================

    print("\n========== ADAPTIVE THRESHOLD ==========")

    print(
        f"Historical threshold: "
        f"{threshold_val:.6f}"
    )

    # ============================================================
    # 11. Display chronological split
    # ============================================================

    print("\n========== TIME-AWARE SPLIT ==========")

    print(
        f"Baseline windows: "
        f"{len(baseline)}"
    )

    print(
        f"Test windows: "
        f"{len(test)}"
    )

    print("\nBaseline period:")

    print(
        baseline["timestamp"].iloc[0]
    )

    print("to")

    print(
        baseline["timestamp"].iloc[-1]
    )

    print("\nTest period:")

    print(
        test["timestamp"].iloc[0]
    )

    print("to")

    print(
        test["timestamp"].iloc[-1]
    )

    # ============================================================
    # 12. Load ground truth
    # ============================================================

    truth = pd.read_csv(
        GROUND_TRUTH_FILE
    )

    truth["timestamp"] = pd.to_datetime(
        truth["timestamp"]
    )

    # ============================================================
    # 13. Match ground truth with test windows
    # ============================================================

    test = test.merge(
        truth,
        on="timestamp",
        how="left",
    )

    # ============================================================
    # 14. Create actual incident labels
    # ============================================================

    test["actual_incident"] = (
        test["incident_type"] != "normal"
    )

    # ============================================================
    # 15. Convert predictions to integers
    # ============================================================

    y_true = test["actual_incident"].astype(int)

    y_pred = test["adaptive_anomaly"].astype(int)

    # ============================================================
    # 16. Evaluate
    # ============================================================

    print("\n========== TIME-AWARE EVALUATION ==========")

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            y_pred,
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Normal",
                "Incident",
            ],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()