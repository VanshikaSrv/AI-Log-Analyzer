import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(
    predictions: pd.DataFrame,
    ground_truth_path: str,
) -> dict:

    truth = pd.read_csv(ground_truth_path)

    truth["timestamp"] = pd.to_datetime(truth["timestamp"])

    results = predictions.merge(
        truth,
        on="timestamp",
        how="left",
    )

    # Ground truth:
    # 0 = normal
    # 1 = actual incident
    results["actual_anomaly"] = (
        results["incident_type"] != "normal"
    ).astype(int)

    # Model:
    # -1 = anomaly
    # 1 = normal
    results["predicted_anomaly"] = (
        results["prediction"] == -1
    ).astype(int)

    y_true = results["actual_anomaly"]
    y_pred = results["predicted_anomaly"]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    metrics = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Normal", "Incident"],
            zero_division=0,
        )
    )

    return metrics