import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


SERVICES = ["API", "AUTH", "DATABASE", "STORAGE"]


NORMAL_MESSAGES = {
    "API": [
        "Request processed successfully",
        "Request completed successfully",
        "Response sent successfully",
    ],
    "AUTH": [
        "User authenticated successfully",
        "Login successful",
        "Token validated successfully",
    ],
    "DATABASE": [
        "Query completed successfully",
        "Database connection healthy",
        "Transaction completed successfully",
    ],
    "STORAGE": [
        "File written successfully",
        "Storage operation completed",
        "File read successfully",
    ],
}


INCIDENTS = {
    "database_failure": {
        "service": "DATABASE",
        "messages": [
            "Connection timeout",
            "Database query failed",
            "Retrying connection",
        ],
    },
    "authentication_failure": {
        "service": "AUTH",
        "messages": [
            "Authentication failed",
            "Invalid credentials",
            "Token validation failed",
        ],
    },
    "api_degradation": {
        "service": "API",
        "messages": [
            "Slow API response",
            "Request timeout",
            "High response latency",
        ],
    },
    "storage_failure": {
        "service": "STORAGE",
        "messages": [
            "Disk write failed",
            "Storage unavailable",
            "File operation failed",
        ],
    },
}


def generate_normal_log(timestamp):

    service = random.choice(SERVICES)

    level = "INFO"

    if random.random() < 0.08:
        level = "WARN"

    message = random.choice(NORMAL_MESSAGES[service])

    # Normal API response time
    if service == "API":
        response_time = random.randint(80, 200)

        return (
            f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
            f"{level} "
            f"{service} "
            f"{message} "
            f"response_time={response_time}"
        )

    return (
        f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
        f"{level} "
        f"{service} "
        f"{message}"
    )


def generate_incident_log(timestamp, incident_type):

    incident = INCIDENTS[incident_type]

    service = incident["service"]

    probability = random.random()

    if probability < 0.65:
        level = "ERROR"
    elif probability < 0.85:
        level = "WARN"
    else:
        level = "INFO"

    message = random.choice(incident["messages"])

    # API degradation
    if incident_type == "api_degradation":

        response_time = random.randint(700, 1500)

        return (
            f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
            f"{level} "
            f"{service} "
            f"{message} "
            f"response_time={response_time}"
        )

    # Authentication failure
    if incident_type == "authentication_failure":

        return (
            f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
            f"{level} "
            f"{service} "
            f"{message} "
            f"auth_failure=1"
        )

    # Storage failure
    if incident_type == "storage_failure":

        return (
            f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
            f"{level} "
            f"{service} "
            f"{message} "
            f"storage_failure=1"
        )

    # Database failure
    if incident_type == "database_failure":

        retry = random.choice([0, 1])

        return (
            f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
            f"{level} "
            f"{service} "
            f"{message} "
            f"retry={retry}"
        )

    return (
        f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
        f"{level} "
        f"{service} "
        f"{message}"
    )


def generate_logs(
    total_logs: int = 10000,
    output_path: str = "data/raw/generated_application_test2.log",
    ground_truth_path: str = "data/processed/ground_truth_test2.csv",
):

    # Fixed seed makes Test 2 reproducible
    random.seed(123)

    # Different start time from Test 1
    start_time = datetime(2026, 10, 1, 10, 0, 0)

    logs = []
    ground_truth = []

    # Different incident timing and duration
    incident_schedule = [
        (1200, 1650, "authentication_failure"),
        (2600, 3200, "api_degradation"),
        (4700, 5050, "database_failure"),
        (6500, 7200, "storage_failure"),
    ]

    for i in range(total_logs):

        timestamp = start_time + timedelta(seconds=i * 5)

        incident_type = "normal"

        for start, end, name in incident_schedule:

            if start <= i < end:
                incident_type = name
                break

        # Slightly more difficult/noisy incident generation
        if incident_type != "normal" and random.random() < 0.70:

            log_line = generate_incident_log(
                timestamp,
                incident_type,
            )

        else:

            log_line = generate_normal_log(timestamp)

        logs.append(log_line)

        ground_truth.append(
            {
                "timestamp": timestamp,
                "incident_type": incident_type,
            }
        )

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(ground_truth_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as file:

        for log in logs:
            file.write(log + "\n")

    truth_df = pd.DataFrame(ground_truth)

    truth_df.to_csv(
        ground_truth_path,
        index=False,
    )

    print(f"Generated {len(logs)} logs")
    print(f"Logs saved to: {output_path}")
    print(f"Ground truth saved to: {ground_truth_path}")

if __name__ == "__main__":
    generate_logs()