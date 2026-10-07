from src.ingestion.loader import load_log_file
from src.parsing.parser import parse_log_line
from src.preprocessing.dataframe_builder import build_dataframe
from src.features.feature_engineering import create_features


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

    print("Number of feature windows:", len(features))

    print("\nFeature columns:")
    print(features.columns.tolist())

    print("\nFirst 5 feature rows:")
    print(features.head())

    print("\nResponse-time statistics:")
    print(
        features[
            [
                "avg_response_time",
                "max_response_time",
                "slow_requests"
            ]
        ].describe()
    )

    print("\nTotal auth failures:")
    print(features["auth_failures"].sum())

    print("\nTotal storage failures:")
    print(features["storage_failures"].sum())

    print("\nTotal retries:")
    print(features["retry_count"].sum())

    print("\nFeature engineering test completed!")


if __name__ == "__main__":
    main()