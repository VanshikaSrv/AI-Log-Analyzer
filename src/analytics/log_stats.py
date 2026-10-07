import pandas as pd

def basic_log(df: pd.DataFrame)->dict:
    if df.empty:
        return{
            "total_logs":0,
            "error_count":0,
            "warn_count":0,
            "info_count":0,
        }
    level_counts=df["level"].value_counts()
    return{
        "total_logs":len(df),
        "error_count":int(level_counts.get("ERROR",0)),
        "warn_count":int(level_counts.get("WARN",0)),
        "info_count":int(level_counts.get("INFO",0)),
    }

def get_top_messages(
    df: pd.DataFrame,
    limit: int = 10
) -> pd.Series:
    """
    Return the most frequent log messages.
    """

    if df.empty:
        return pd.Series(dtype="int64")

    return df["message"].value_counts().head(limit)
def get_error_counts_over_time(df:pd.DataFrame,freq: str="5min")->pd.Series:
    if df.empty:
        return pd.Series(dtype="int64")
    error=df[df["level"]=="ERROR"].copy()
    if error.empty:
        return pd.Series(dtype="int64")
    error=error.set_index("timestamp")
    return error["level"].resample(freq).count()