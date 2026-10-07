from src.pipeline.log_pipeline import process_log_file
from src.analytics.log_stats import (
    basic_log,
    get_top_messages,
    get_error_counts_over_time,
)


df = process_log_file("data/raw/application.log")

print("\n===== DATA =====")
print(df)

print("\n===== BASIC STATISTICS =====")
print(basic_log(df))

print("\n===== TOP MESSAGES =====")
print(get_top_messages(df))

print("\n===== ERRORS OVER TIME =====")
print(get_error_counts_over_time(df))