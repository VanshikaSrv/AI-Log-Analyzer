from src.ingestion.loader import load_log_file
from src.parsing.parser import parse_log_line
from src.preprocessing.dataframe_builder import build_dataframe
from src.features.feature_engineering import create_features
from src.detection.anomaly_detector import detect_anomalies


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
    # 3. Detect anomalies
    # -------------------------

    predictions = detect_anomalies(
        features,
        contamination=0.05
    )

    # -------------------------
    # 4. Load ground truth
    # -------------------------

    truth = __import__("pandas").read_csv(
        GROUND_TRUTH_FILE
    )

    truth["timestamp"] = __import__("pandas").to_datetime(
        truth["timestamp"]
    )

    # -------------------------
    # 5. Merge predictions
    # -------------------------

    results = predictions.merge(
        truth,
        on="timestamp",
        how="left"
    )

    results["actual_anomaly"] = (
        results["incident_type"] != "normal"
    )

    # -------------------------
    # 6. Show detected anomalies
    # -------------------------

    anomalies = results[
        results["prediction"] == -1
    ]

    print("\nDetected anomaly windows:")
    print(
        anomalies[
            [
                "timestamp",
                "incident_type",
                "prediction",
                "anomaly_score",
                "error_count",
                "retry_count",
                "auth_failures",
                "storage_failures",
                "avg_response_time",
                "max_response_time",
                "slow_requests"
            ]
        ].to_string(index=False)
    )

    # -------------------------
    # 7. Incident detection count
    # -------------------------

    print("\nDetected anomalies by incident type:")

    print(
        anomalies["incident_type"]
        .value_counts()
    )

    # -------------------------
    # 8. Actual incident windows
    # -------------------------

    print("\nActual incident windows:")

    print(
        results[
            results["actual_anomaly"]
        ]["incident_type"]
        .value_counts()
    )


if __name__ == "__main__":
    main()