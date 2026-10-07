import pandas as pd
from sklearn.ensemble import IsolationForest


FEATURE_COLUMNS = [
    "total_logs",
    "error_count",
    "warning_count",
    "info_count",

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


def fit_detector(features: pd.DataFrame, contamination: float = 0.05):
    if features.empty:
        raise ValueError("Cannot train detector on empty data.")

    X = features[FEATURE_COLUMNS]

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )

    model.fit(X)

    return model


def score_data(
    model,
    features: pd.DataFrame
) -> pd.DataFrame:

    if features.empty:
        return features.copy()

    data = features.copy()
    X = data[FEATURE_COLUMNS]

    data["prediction"] = model.predict(X)
    data["anomaly_score"] = model.decision_function(X)

    return data