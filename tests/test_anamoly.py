from src.ingestion.loader import load_log_file
from src.parsing.parser import parse_log_line
from src.preprocessing.dataframe_builder import build_dataframe
from src.features.feature_engineering import create_features
from src.detection.anomaly_detector import detect_anomalies, FEATURE_COLUMNS


def main():

    lines = load_log_file(
        "data/raw/generated_application.log"
    )

    events = []

    for line in lines:
        event = parse_log_line(line)

        if event is not None:
            events.append(event)

    df = build_dataframe(events)

    features = create_features(df)

    print("Number of features used by model:", len(FEATURE_COLUMNS))

    print("\nFeatures used by Isolation Forest:")
    for feature in FEATURE_COLUMNS:
        print("-", feature)

    predictions = detect_anomalies(features)

    print("\nPrediction counts:")
    print(predictions["prediction"].value_counts())

    print("\nAnomaly score range:")
    print(
        "Minimum:",
        predictions["anomaly_score"].min()
    )

    print(
        "Maximum:",
        predictions["anomaly_score"].max()
    )

    print("\nAnomaly detector test completed!")


if __name__ == "__main__":
    main()