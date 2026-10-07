from src.ingestion.loader import load_log_file
from src.parsing.parser import parse_log_line
from src.preprocessing.dataframe_builder import build_dataframe
from src.features.feature_engineering import create_features
from src.detection.anomaly_detector import detect_anomalies
from src.evaluation.model_evaluation import evaluate_model


LOG_FILE = "data/raw/generated_application.log"
GROUND_TRUTH_FILE = "data/processed/ground_truth.csv"


def main():

    # -------------------------
    # 1. Load raw logs
    # -------------------------

    lines = load_log_file(LOG_FILE)

    print("Total log lines:", len(lines))

    # -------------------------
    # 2. Parse logs
    # -------------------------

    events = []

    for line in lines:
        event = parse_log_line(line)

        if event is not None:
            events.append(event)

    print("Successfully parsed:", len(events))

    # -------------------------
    # 3. Build DataFrame
    # -------------------------

    df = build_dataframe(events)

    print("DataFrame rows:", len(df))

    # -------------------------
    # 4. Create ML features
    # -------------------------

    features = create_features(df)

    print("Feature windows:", len(features))

    # -------------------------
    # 5. Detect anomalies
    # -------------------------

    predictions = detect_anomalies(
        features,
        contamination=0.20
    )

    print(
        "\nPredictions:"
    )
    print(
        predictions["prediction"].value_counts()
    )

    # -------------------------
    # 6. Evaluate predictions
    # -------------------------

    metrics = evaluate_model(
        predictions,
        GROUND_TRUTH_FILE
    )

    # -------------------------
    # 7. Print metrics
    # -------------------------

    print("\nMetrics:")
    print(metrics)


if __name__ == "__main__":
    main()