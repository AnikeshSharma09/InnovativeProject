"""
Enterprise-Grade Sentence Construction & NLP Autocorrect Engine.
Supports:
1. Real-time character accumulation with pause-based auto-spacing.
2. Robust typo recovery (Levenshtein distance) for hands that tremble/mis-key letters.
3. Contextual next-word prediction & autocomplete (corporate/MNC domain).
4. Whole-phrase integration (merging whole-sentence signs into text stream).
5. Enterprise exports: Daily Standup format, Jira comments, and timestamped transcripts.
"""
import time
from typing import List, Tuple, Dict, Optional


# Professional Workplace & General English Dictionary for Autocorrect & Suggestions
ENTERPRISE_DICTIONARY = [
    # Meeting & Communication
    "HELLO", "GREETINGS", "THANKS", "THANK", "YOU", "WELCOME", "PLEASE", "SORRY", "EXCUSE",
    "YES", "NO", "AGREE", "DISAGREE", "APPROVED", "APPROVAL", "CONFIRMED", "WAIT", "HELP",
    "QUESTION", "ANSWER", "DISCUSS", "MEETING", "STANDUP", "SYNC", "CALL", "MINUTES",
    "AGENDA", "NOTES", "TOPIC", "RECORD", "SHARE", "SCREEN", "MUTE", "UNMUTE", "HEAR",
    # Project Management & Roles
    "PROJECT", "TASK", "TICKET", "SPRINT", "BACKLOG", "STORY", "FEATURE", "BUG", "ISSUE",
    "BLOCKER", "PROGRESS", "STATUS", "UPDATE", "REPORT", "DEADLINE", "TIMELINE", "ROADMAP",
    "DEPLOY", "RELEASE", "BUILD", "TEST", "CODE", "REVIEW", "MERGE", "BRANCH", "PIPELINE",
    "SERVER", "DATABASE", "FRONTEND", "BACKEND", "API", "CLOUD", "CLIENT", "TEAM", "LEAD",
    # Conversational Phrases & Connectors
    "I", "WE", "YOU", "THEY", "THIS", "THAT", "WHAT", "WHEN", "WHERE", "HOW", "WHY",
    "IS", "ARE", "WAS", "WERE", "WILL", "HAVE", "HAD", "NEED", "WANT", "DO", "CAN",
    "WORK", "WORKING", "DONE", "FINISHED", "READY", "NEXT", "TODAY", "YESTERDAY", "TOMORROW",
    "GOOD", "GREAT", "FINE", "OKAY", "CLEAR", "UNDERSTAND", "ASSISTANCE", "IMPORTANT"
]

# Contextual Bigrams for Corporate/MNC Autocomplete Suggestions
CORPORATE_BIGRAMS: Dict[str, List[str]] = {
    "I": ["AGREE", "THINK", "NEED", "WILL", "HAVE", "AM"],
    "WE": ["NEED", "SHOULD", "CAN", "WILL", "HAVE", "DISCUSS"],
    "THANK": ["YOU", "VERY", "EVERYONE"],
    "THANKS": ["TEAM", "EVERYONE", "AGAIN"],
    "PLEASE": ["WAIT", "REVIEW", "HELP", "CONFIRM", "CHECK"],
    "PROJECT": ["STATUS", "UPDATE", "ROADMAP", "DEADLINE", "SCOPE"],
    "STANDUP": ["UPDATE", "MEETING", "NOTES", "CALL"],
    "STATUS": ["UPDATE", "REPORT", "OKAY", "READY"],
    "TASK": ["COMPLETED", "ASSIGNED", "BLOCKED", "PROGRESS"],
    "SPRINT": ["BACKLOG", "GOAL", "REVIEW", "PLANNING"],
    "DAILY": ["STANDUP", "SYNC", "UPDATE", "MEETING"],
    "GOOD": ["MORNING", "AFTERNOON", "JOB", "PROGRESS"],
    "CAN": ["YOU", "WE", "HELP", "CONFIRM"],
    "NEED": ["HELP", "TIME", "APPROVAL", "ACCESS", "REVIEW"],
    "CODE": ["REVIEW", "MERGED", "TESTED", "DEPLOYED"],
    "BUG": ["FIXED", "FOUND", "CRITICAL", "BLOCKED"],
    "MEETING": ["NOTES", "MINUTES", "START", "SCHEDULED"]
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes minimum edit operations to transform s1 into s2."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


class EnterpriseSentenceEngine:
    """
    Stateful sentence building and NLP correction engine designed for
    enterprise collaboration and virtual project meetings.
    """

    def __init__(self, pause_autospace_seconds: float = 1.4):
        self.pause_autospace_seconds = pause_autospace_seconds
        
        # Sentence State
        self.current_word: str = ""
        self.committed_words: List[str] = []
        self.meeting_transcript: List[Dict[str, str]] = []
        
        # Timing trackers
        self.last_char_timestamp: float = time.time()
        self.last_committed_char: Optional[str] = None
        
        # Word frequency cache
        self.vocab = set(ENTERPRISE_DICTIONARY)

    def reset(self):
        """Clears current active sentence."""
        self.current_word = ""
        self.committed_words.clear()
        self.last_char_timestamp = time.time()
        self.last_committed_char = None

    def clear_all(self):
        """Clears both sentence and entire meeting transcript."""
        self.reset()
        self.meeting_transcript.clear()

    def add_character(self, char: str) -> None:
        """
        Processes a single recognized character or control sign.
        Handles: normal chars, 'space', 'del', 'nothing'.
        """
        now = time.time()
        self.last_char_timestamp = now

        if char == "nothing":
            return

        if char == "space":
            self.commit_current_word()
            return

        if char == "del":
            self.backspace()
            return

        # Regular alphanumeric character
        self.current_word += char.upper()

    def add_phrase(self, phrase: str) -> None:
        """
        Directly commits an entire recognized sign language sentence/phrase
        (e.g., 'THANK YOU', 'I AGREE', 'PLEASE WAIT').
        """
        # If there was a partially typed word, commit it first
        if self.current_word:
            self.commit_current_word()

        clean_phrase = phrase.strip().upper()
        words = clean_phrase.split()
        self.committed_words.extend(words)

        # Log entry in meeting transcript
        entry_time = time.strftime("%H:%M:%S")
        self.meeting_transcript.append({
            "timestamp": entry_time,
            "speaker": "ASL Presenter",
            "text": clean_phrase,
            "type": "phrase"
        })
        self.last_char_timestamp = time.time()

    def commit_current_word(self, autocorrect: bool = True) -> Optional[str]:
        """
        Commits the current word buffer to the sentence, optionally
        applying intelligent typo autocorrection.
        """
        if not self.current_word:
            return None

        word = self.current_word.strip().upper()
        final_word = word

        if autocorrect and len(word) >= 3:
            # Check if exact match exists in vocabulary
            if word not in self.vocab:
                suggestions = self.get_autocorrect_suggestions(word, max_suggestions=1)
                if suggestions and suggestions[0][1] <= 2:  # Levenshtein distance <= 2
                    final_word = suggestions[0][0]

        self.committed_words.append(final_word)
        self.current_word = ""

        # Update transcript
        entry_time = time.strftime("%H:%M:%S")
        self.meeting_transcript.append({
            "timestamp": entry_time,
            "speaker": "ASL Presenter",
            "text": final_word,
            "type": "word"
        })
        return final_word

    def backspace(self) -> None:
        """Deletes last typed character or last committed word."""
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.committed_words:
            self.current_word = self.committed_words.pop()

    def check_pause_autospace(self) -> bool:
        """
        Auto-commits current word if user paused signing for configured duration.
        """
        if self.current_word and (time.time() - self.last_char_timestamp) > self.pause_autospace_seconds:
            self.commit_current_word(autocorrect=True)
            return True
        return False

    def get_full_sentence(self, include_current: bool = True) -> str:
        """Returns the full accumulated sentence formatted nicely."""
        words = list(self.committed_words)
        if include_current and self.current_word:
            words.append(f"{self.current_word}...")
        
        raw_str = " ".join(words)
        if not raw_str:
            return ""

        # Capitalize sentence
        return raw_str[0].upper() + raw_str[1:] if len(raw_str) > 1 else raw_str.upper()

    def get_autocorrect_suggestions(self, word: Optional[str] = None, max_suggestions: int = 3) -> List[Tuple[str, int]]:
        """
        Calculates closest dictionary words using Levenshtein edit-distance.
        Returns list of (suggested_word, edit_distance).
        """
        target = (word or self.current_word).strip().upper()
        if not target or len(target) < 2:
            return []

        scored = []
        for v in ENTERPRISE_DICTIONARY:
            dist = levenshtein_distance(target, v)
            if dist <= 2:
                scored.append((v, dist))

        scored.sort(key=lambda item: item[1])
        return scored[:max_suggestions]

    def get_next_word_predictions(self, max_predictions: int = 3) -> List[str]:
        """
        Predicts next word based on the last committed word in corporate context.
        """
        if not self.committed_words:
            return ["I", "STATUS", "THANK", "STANDUP"]

        last_word = self.committed_words[-1].upper()
        if last_word in CORPORATE_BIGRAMS:
            return CORPORATE_BIGRAMS[last_word][:max_predictions]

        return ["UPDATE", "AGREE", "PLEASE", "REPORT"]

    def apply_suggestion(self, suggestion: str) -> None:
        """Directly inserts a clicked suggestion word."""
        self.current_word = ""
        self.committed_words.append(suggestion.upper())
        self.last_char_timestamp = time.time()

    # =========================================================================
    # ENTERPRISE / MNC PROJECT MANAGEMENT EXPORT FORMATTERS
    # =========================================================================

    def format_daily_standup(self) -> str:
        """Formats accumulated sentence into a professional Daily Standup update."""
        sentence = self.get_full_sentence(include_current=True) or "Worked on task implementation and testing."
        date_str = time.strftime("%Y-%m-%d")
        return (
            f"**Daily Standup Update** - {date_str}\n"
            f"----------------------------------------\n"
            f"* **Today's Focus & Update:** {sentence}\n"
            f"* **Status:** Active / In-Progress\n"
            f"* **Blockers:** None reported\n"
            f"*(Generated via Motion-Aware Sign Language AI)*"
        )

    def format_jira_comment(self, ticket_key: str = "PROJ-104") -> str:
        """Formats accumulated sentence into an official Jira / Trello ticket update."""
        sentence = self.get_full_sentence(include_current=True) or "Completed milestone validation."
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"**[{ticket_key}] Work Progress Comment** ({timestamp})\n"
            f"------------------------------------------------\n"
            f"{sentence}\n\n"
            f"*Signed & translated via AI Gesture Suite*"
        )

    def export_transcript_markdown(self) -> str:
        """Exports meeting transcript with timestamps for Slack / Teams / Zoom minutes."""
        if not self.meeting_transcript:
            return f"# Virtual Meeting Minutes\n*Date: {time.strftime('%Y-%m-%d %H:%M')}*\n\n*(No signed transcript recorded yet.)*"

        lines = [
            f"# Virtual Meeting Live Captions & Minutes",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Modality:** Motion-Aware ASL Real-Time Translation",
            f"",
            f"| Time | Speaker | Recognized Sign Language Content |",
            f"| :--- | :--- | :--- |"
        ]
        for item in self.meeting_transcript:
            lines.append(f"| `{item['timestamp']}` | **{item['speaker']}** | {item['text']} |")

        lines.append("")
        lines.append(f"**Final Consolidated Sentence:** {self.get_full_sentence(include_current=False)}")
        return "\n".join(lines)
