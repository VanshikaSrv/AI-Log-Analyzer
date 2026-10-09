import pandas as pd


DIAGNOSTIC_FEATURES = [
    "database_errors",
    "auth_errors",
    "api_errors",
    "storage_errors",
    "retry_count",
    "auth_failures",
    "storage_failures",
    "avg_response_time",
    "max_response_time",
    "slow_requests",
    "error_rate",
]


def calculate_baseline_statistics(
    baseline_features: pd.DataFrame
) -> pd.DataFrame:

    if baseline_features.empty:
        raise ValueError(
            "Cannot calculate baseline from empty data."
        )

    statistics = baseline_features[
        DIAGNOSTIC_FEATURES
    ].agg(["mean", "std"]).T

    statistics = statistics.fillna(0)

    return statistics