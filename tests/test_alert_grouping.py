import pandas as pd

from src.incidents.alert_grouping import group_alerts


def main():

    # ==========================================
    # TEST 1: Multiple incident groups
    # ==========================================

    predictions = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-10-02 19:40:00",
            "2026-10-02 19:45:00",
            "2026-10-02 19:50:00",
            "2026-10-02 20:05:00",
            "2026-10-02 20:10:00",
        ]),
        "adaptive_anomaly": [
            True,
            True,
            True,
            True,
            True,
        ]
    })

    incidents = group_alerts(predictions)

    print("\n========== TEST 1 ==========")
    print(incidents)


    # ==========================================
    # TEST 2: False value between alerts
    # ==========================================

    predictions = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-10-02 19:40:00",
            "2026-10-02 19:45:00",
            "2026-10-02 19:50:00",
        ]),
        "adaptive_anomaly": [
            True,
            False,
            True,
        ]
    })

    incidents = group_alerts(predictions)

    print("\n========== TEST 2 ==========")
    print(incidents)


    # ==========================================
    # TEST 3: No alerts
    # ==========================================

    predictions = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-10-02 19:40:00",
            "2026-10-02 19:45:00",
            "2026-10-02 19:50:00",
        ]),
        "adaptive_anomaly": [
            False,
            False,
            False,
        ]
    })

    incidents = group_alerts(predictions)

    print("\n========== TEST 3 ==========")
    print(incidents)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()