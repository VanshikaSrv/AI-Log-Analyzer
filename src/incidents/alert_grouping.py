import pandas as pd


def group_alerts(
    predictions: pd.DataFrame,
    max_gap_minutes: int = 5,
) -> pd.DataFrame:

    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "incident_id",
                "start_time",
                "end_time",
                "alert_count",
            ]
        )

    data = predictions[
        predictions["adaptive_anomaly"] == True
    ].copy()

    if data.empty:
        return pd.DataFrame(
            columns=[
                "incident_id",
                "start_time",
                "end_time",
                "alert_count",
            ]
        )

    data = data.sort_values("timestamp").reset_index(drop=True)

    time_gap = (
        data["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    new_incident = (
        time_gap.isna()
        | (time_gap > max_gap_minutes)
    )

    data["incident_id"] = new_incident.cumsum()

    incidents = (
        data.groupby("incident_id")
        .agg(
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            alert_count=("timestamp", "count"),
        )
        .reset_index()
    )

    return incidents