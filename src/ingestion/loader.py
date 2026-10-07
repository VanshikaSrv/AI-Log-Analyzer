from pathlib import Path

def load_log_file(file_path: str)->list[str]:
    PATH=Path("C:\\ai-log-analyzer\\data\\raw\\generated_application_test2.log")
    if not Path.exists(PATH):
        raise FileNotFoundError(f"File not found: {PATH}")
    else:
        with open(PATH, 'r') as file:
            lines = file.readlines()
    return lines