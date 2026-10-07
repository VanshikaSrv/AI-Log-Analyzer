import pandas as pd

def build_dataframe(events: list[dict])->pd.DataFrame:
    """Conver logs into dataframe """
    df=pd.DataFrame(events)
    if df.empty:
        return df
    df["timestamp"]=pd.to_datetime(df["timestamp"])
    return df