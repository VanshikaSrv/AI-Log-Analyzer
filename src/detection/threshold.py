import pandas as pd


def calculate_baseline_scores(
    scores: pd.Series
) -> pd.Series:

    if scores.empty:
        raise ValueError("Cannot create baseline from empty scores.")

    return scores.copy()


def calculate_adaptive_threshold(
    scores: pd.Series,
    k: float = 3.0
) -> float:

    if scores.empty:
        raise ValueError("Cannot calculate threshold from empty scores.")

    median = scores.median()

    absolute_deviation = (scores - median).abs()

    mad = absolute_deviation.median()

    threshold = median - (k * mad)

    return threshold


def apply_threshold(
    scores: pd.Series,
    threshold: float
) -> pd.Series:

    return scores < threshold