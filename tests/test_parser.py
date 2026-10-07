from src.ingestion.loader import load_log_file
from src.parsing.parser import parse_log_line


LOG_FILE = "data/raw/generated_application.log"


def main():
    lines = load_log_file(LOG_FILE)

    parsed_events = []

    for line in lines:
        event = parse_log_line(line)

        if event is not None:
            parsed_events.append(event)

    print("Total log lines:", len(lines))
    print("Successfully parsed:", len(parsed_events))

    # Convert to a simple list of values
    response_times = [
        event["response_time"]
        for event in parsed_events
        if event["response_time"] > 0
    ]

    auth_failures = [
        event
        for event in parsed_events
        if event["auth_failure"] == 1
    ]

    storage_failures = [
        event
        for event in parsed_events
        if event["storage_failure"] == 1
    ]

    retries = [
        event
        for event in parsed_events
        if event["retry"] == 1
    ]

    print("Logs with response_time:", len(response_times))
    print("Auth failures:", len(auth_failures))
    print("Storage failures:", len(storage_failures))
    print("Retries:", len(retries))

    print("\nSample parsed events:")

    for event in parsed_events[:5]:
        print(event)

    print("\nParser integration check completed!")


if __name__ == "__main__":
    main()