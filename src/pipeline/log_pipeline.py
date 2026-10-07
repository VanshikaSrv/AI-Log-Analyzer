from src.ingestion.loader import load_log_file
from src.parsing.parser import parse_log_line
from src.preprocessing.dataframe_builder import build_dataframe

def process_log_file(file_path : str):
      """
    Load a log file and parse all valid log lines.

    Returns:
        A pandas DataFrame containing the structured log events.
    """
      lines=load_log_file(file_path)
      events=[]
      for line in lines:
            event=parse_log_line(line)
            if event is not None:
                  events.append(event)
      return build_dataframe(events)
