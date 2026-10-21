"""
Utility for saving real-time predicted signs, words, and sentences to a persistent file.
"""
import os
import time
from pathlib import Path
from typing import List


DEFAULT_LOG_FILE = Path("data/predicted_signs.txt")


class PredictionLogger:
    """
    Manages persistent logging of recognized signs, words, and sentences
    to a text file with timestamps.
    """

    def __init__(self, file_path: Path = DEFAULT_LOG_FILE):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(f"# Real-Time Sign Language Predicted Text Log\n# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n# ------------------------------------------------------------\n")

    def log_prediction(self, text: str, tag: str = "PREDICTION") -> None:
        """Appends a predicted word, phrase, or sentence with timestamp."""
        cleaned = text.strip()
        if not cleaned:
            return

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{tag}] {cleaned}\n"

        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(entry)

    def get_file_content(self) -> str:
        """Reads and returns the complete log content."""
        if not self.file_path.exists():
            return ""
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()

    def get_recent_entries(self, n: int = 15) -> List[str]:
        """Returns the last n logged prediction lines."""
        content = self.get_file_content()
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]
        return lines[-n:]

    def clear(self) -> None:
        """Clears the file and writes a fresh header."""
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(f"# Real-Time Sign Language Predicted Text Log\n# Reset: {time.strftime('%Y-%m-%d %H:%M:%S')}\n# ------------------------------------------------------------\n")
