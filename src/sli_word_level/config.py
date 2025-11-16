"""
Configuration for the SLI Word-Level real-time interpreter.
"""

from pathlib import Path
import os

# Root of the project (assumes this file is inside src/sli_word_level/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default path to the trained word-level model
MODEL_PATH = Path(
    os.getenv("SLI_MODEL_PATH", PROJECT_ROOT / "models" / "SLI_V2.keras")
)

# Sliding-window configuration
WINDOW_SIZE: int = 30
MIN_FRAMES: int = 20
CONFIDENCE_THRESHOLD: float = 0.80

# Target glosses (word classes) – adjust if you add more
TARGET_GLOSSES = ["any", "thank you", "bye", "question"]
FOLDER_TO_LABEL = {gloss: i for i, gloss in enumerate(TARGET_GLOSSES)}
LABEL_TO_FOLDER = {i: gloss for gloss, i in FOLDER_TO_LABEL.items()}