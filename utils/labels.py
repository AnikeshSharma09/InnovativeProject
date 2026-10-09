"""
ASL Alphabet label definitions, mappings, and display helpers.
Supports standard 26 letters (A-Z) and standard special tokens (space, del, nothing).
"""
import string

# Standard 10 Digits
DIGIT_CLASSES = [str(i) for i in range(10)]

# Standard 26 English Alphabet letters
ALPHABET_LETTERS = list(string.ascii_uppercase)

# Standard ASL special action classes common in ASL datasets
SPECIAL_CLASSES = ["del", "nothing", "space"]

# Complete 36+ class ASL set (Digits 0-9, Letters A-Z, Special Tokens)
DEFAULT_ASL_CLASSES = sorted(DIGIT_CLASSES + ALPHABET_LETTERS + SPECIAL_CLASSES)

# Human-readable display mapping
LABEL_DISPLAY_MAP = {
    "del": "[DEL]",
    "nothing": "[NO HAND]",
    "space": "[SPACE]"
}
for d in DIGIT_CLASSES:
    LABEL_DISPLAY_MAP[d] = f"Digit {d}"
for char in ALPHABET_LETTERS:
    LABEL_DISPLAY_MAP[char] = char


def format_label_for_display(label: str) -> str:
    """Returns a clean display name for a given class label."""
    if not label:
        return "N/A"
    return LABEL_DISPLAY_MAP.get(str(label).strip(), str(label).upper())


def is_valid_asl_label(label: str) -> bool:
    """Checks if a given string is a valid ASL label."""
    if not label:
        return False
    norm = str(label).strip()
    return norm in DIGIT_CLASSES or norm.upper() in ALPHABET_LETTERS or norm.lower() in SPECIAL_CLASSES
