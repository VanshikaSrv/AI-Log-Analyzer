import pandas as pd


def create_features(
    df: pd.DataFrame,
    frequency: str = "5min"
) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()

    data = df.copy()
    data = data.set_index("timestamp")

    # -------------------------
    # Basic log volume features
    # -------------------------

    total_logs = data.resample(frequency).size()

    error_count = (
        data["level"]
        .eq("ERROR")
        .resample(frequency)
        .sum()
    )

    warning_count = (
        data["level"]
        .eq("WARN")
        .resample(frequency)
        .sum()
    )

    info_count = (
        data["level"]
        .eq("INFO")
        .resample(frequency)
        .sum()
    )

    # -------------------------
    # Service-specific errors
    # -------------------------

    database_errors = (
        ((data["level"] == "ERROR") &
         (data["service"] == "DATABASE"))
        .resample(frequency)
        .sum()
    )

    auth_errors = (
        ((data["level"] == "ERROR") &
         (data["service"] == "AUTH"))
        .resample(frequency)
        .sum()
    )

    api_errors = (
        ((data["level"] == "ERROR") &
         (data["service"] == "API"))
        .resample(frequency)
        .sum()
    )

    storage_errors = (
        ((data["level"] == "ERROR") &
         (data["service"] == "STORAGE"))
        .resample(frequency)
        .sum()
    )

    # -------------------------
    # Structured incident signals
    # -------------------------

    retry_count = (
        data["retry"]
        .resample(frequency)
        .sum()
    )

    auth_failures = (
        data["auth_failure"]
        .resample(frequency)
        .sum()
    )

    storage_failures = (
        data["storage_failure"]
        .resample(frequency)
        .sum()
    )

    # -------------------------
    # API response-time features
    # -------------------------

    api_response_times = data["response_time"].where(
        data["response_time"] > 0
    )

    avg_response_time = (
        api_response_times
        .resample(frequency)
        .mean()
    )

    max_response_time = (
        api_response_times
        .resample(frequency)
        .max()
    )

    slow_requests = (
        (data["response_time"] > 500)
        .resample(frequency)
        .sum()
    )

    # -------------------------
    # Combine features
    # -------------------------

    features = pd.DataFrame({
        "total_logs": total_logs,
        "error_count": error_count,
        "warning_count": warning_count,
        "info_count": info_count,

        "database_errors": database_errors,
        "auth_errors": auth_errors,
        "api_errors": api_errors,
        "storage_errors": storage_errors,

        "retry_count": retry_count,
        "auth_failures": auth_failures,
        "storage_failures": storage_failures,

        "avg_response_time": avg_response_time,
        "max_response_time": max_response_time,
        "slow_requests": slow_requests,
    })

    # -------------------------
    # Error rate
    # -------------------------

    features["error_rate"] = (
        features["error_count"] /
        features["total_logs"]
    )

    features["error_rate"] = features["error_rate"].fillna(0)

    # Missing response-time values mean
    # that no API response was recorded.
    features["avg_response_time"] = (
        features["avg_response_time"]
        .fillna(0)
    )

    features["max_response_time"] = (
        features["max_response_time"]
        .fillna(0)
    )

    return features.reset_index()