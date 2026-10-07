import re


LOG_PATTERN = (
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} "
    r"\d{2}:\d{2}:\d{2}) "
    r"(?P<level>INFO|WARN|ERROR|CRITICAL|DEBUG) "
    r"(?P<service>[A-Z]+) "
    r"(?P<message>.*?)"
    r"(?: response_time=(?P<response_time>\d+))?"
    r"(?: auth_failure=(?P<auth_failure>\d+))?"
    r"(?: storage_failure=(?P<storage_failure>\d+))?"
    r"(?: retry=(?P<retry>\d+))?$"
)


def parse_log_line(line: str) -> dict | None:

    match = re.match(LOG_PATTERN, line)

    if not match:
        return None

    data = match.groupdict()

    data["response_time"] = int(data["response_time"] or 0)
    data["auth_failure"] = int(data["auth_failure"] or 0)
    data["storage_failure"] = int(data["storage_failure"] or 0)
    data["retry"] = int(data["retry"] or 0)

    return data

