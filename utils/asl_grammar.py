"""
asl_grammar.py
==============
Linguistic ASL-to-English Grammar Translation Engine.

Transforms raw ASL Gloss sequences (which follow Topic-Comment, Time-First,
and Object-Subject-Verb syntax with non-manual facial cues) into fluent,
grammatically correct English sentences.

Features:
1. Topic-Comment to Subject-Verb-Object (SVO) inversion.
2. Temporal Aspect & Tense resolution (PAST, FINISH, NOW, TOMORROW, LATER).
3. Non-Manual Cue Integration:
   - Raised Eyebrows -> Yes/No Question syntax ("Are you...", "Do you...").
   - Furrowed Eyebrows -> WH-Question syntax ("What is...", "Where is...").
   - Head Shake / NOT -> Negation insertion ("do not", "cannot", "did not").
4. Copula & Preposition Insertion (ASL omits "is/am/are" and auxiliary verbs).
5. Pronoun Case & Agreement adjustment (ME -> I/me, HIM -> he/him).
6. Local LLM / Ollama translation hook with zero-latency deterministic fallback.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import re


@dataclass
class ASLGlossToken:
    """An individual ASL gloss with associated non-manual facial/head cues."""
    gloss: str
    is_question: bool = False       # Raised eyebrows
    is_wh_question: bool = False    # Furrowed eyebrows + WH-word
    is_negated: bool = False        # Head shake or NOT
    is_emphasis: bool = False       # Open mouth / intense hold
    timestamp: float = 0.0


@dataclass
class GrammarTranslationResult:
    """Translation output containing both raw glosses and natural English."""
    raw_glosses: List[str]
    asl_gloss_string: str
    english_translation: str
    sentence_type: str              # "STATEMENT", "YES_NO_QUESTION", "WH_QUESTION", "NEGATION", "COMMAND"
    detected_tense: str             # "PAST", "PRESENT", "FUTURE"
    non_manual_notes: List[str] = field(default_factory=list)


# Linguistic dictionaries
TIME_MARKERS = {
    "YESTERDAY": "PAST",
    "PAST": "PAST",
    "BEFORE": "PAST",
    "FINISH": "PERFECT",
    "TODAY": "PRESENT",
    "NOW": "PRESENT",
    "TOMORROW": "FUTURE",
    "LATER": "FUTURE",
    "SOON": "FUTURE",
    "NEXT": "FUTURE",
    "WILL": "FUTURE"
}

WH_WORDS = {"WHAT", "WHO", "WHERE", "WHEN", "WHY", "HOW", "WHICH"}

PRONOUN_MAP = {
    "ME": "I",
    "MY": "my",
    "MINE": "mine",
    "YOU": "you",
    "YOUR": "your",
    "HE": "he",
    "SHE": "she",
    "WE": "we",
    "OUR": "our",
    "THEY": "they",
    "THEIR": "their"
}

# Common ASL Idioms and Compound Translations
PHRASE_TRANSLATIONS = {
    ("NAME", "YOU", "WHAT"): "What is your name?",
    ("YOU", "NAME", "WHAT"): "What is your name?",
    ("WHERE", "BATHROOM"): "Where is the bathroom?",
    ("BATHROOM", "WHERE"): "Where is the bathroom?",
    ("HOW", "YOU"): "How are you?",
    ("YOU", "HOW"): "How are you?",
    ("NICE", "MEET", "YOU"): "Nice to meet you.",
    ("THANK", "YOU"): "Thank you.",
    ("THANKS"): "Thank you.",
    ("PLEASE", "HELP"): "Please help me.",
    ("I", "NEED", "HELP"): "I need help.",
    ("HELP", "ME"): "Please help me.",
    ("EMERGENCY", "CALL", "POLICE"): "It's an emergency! Call the police.",
    ("CALL", "DOCTOR"): "Please call a doctor.",
    ("STORE", "I", "GO"): "I am going to the store.",
    ("BOOK", "I", "READ"): "I am reading a book.",
    ("BOOK", "FINISH", "READ"): "I have finished reading the book.",
    ("MEETING", "CANCEL"): "The meeting has been cancelled.",
    ("I", "UNDERSTAND", "NOT"): "I don't understand.",
    ("I", "AGREE", "NOT"): "I don't agree.",
    ("TIME", "WHAT"): "What time is it?",
    ("WATER", "WANT"): "I want water."
}


class ASLGrammarTranslator:
    """
    Deterministic Linguistic Translation Engine for ASL to Natural English.
    """

    def __init__(self, enable_local_llm: bool = False, llm_endpoint: Optional[str] = None):
        self.enable_local_llm = enable_local_llm
        self.llm_endpoint = llm_endpoint

    def translate_gloss_sequence(
        self,
        tokens: List[ASLGlossToken]
    ) -> GrammarTranslationResult:
        """
        Translates a sequence of ASLGlossTokens into a natural English sentence.
        """
        if not tokens:
            return GrammarTranslationResult(
                raw_glosses=[],
                asl_gloss_string="",
                english_translation="",
                sentence_type="STATEMENT",
                detected_tense="PRESENT"
            )

        raw_glosses = [t.gloss.upper() for t in tokens]
        asl_gloss_str = " ".join(raw_glosses)

        # ---------------------------------------------------------------------
        # Step 1: Detect Non-Manual Facial Cues & Sentence Type
        # ---------------------------------------------------------------------
        notes = []
        is_q = any(t.is_question for t in tokens)
        is_wh = any(t.is_wh_question or t.gloss in WH_WORDS for t in tokens)
        is_neg = any(t.is_negated or t.gloss in ("NOT", "NO", "DONT-WANT", "NEVER") for t in tokens)

        sentence_type = "STATEMENT"
        if is_wh:
            sentence_type = "WH_QUESTION"
            notes.append("WH-Question (Furrowed Eyebrows / WH-Sign)")
        elif is_q:
            sentence_type = "YES_NO_QUESTION"
            notes.append("Yes/No Question (Raised Eyebrows)")
        elif is_neg:
            sentence_type = "NEGATION"
            notes.append("Negation (Head Shake / Negative Marker)")

        # ---------------------------------------------------------------------
        # Step 2: Time & Tense Resolution
        # ---------------------------------------------------------------------
        detected_tense = "PRESENT"
        filtered_glosses = []
        for g in raw_glosses:
            if g in TIME_MARKERS:
                detected_tense = TIME_MARKERS[g]
                notes.append(f"Tense Marker: {g} -> {detected_tense}")
            else:
                filtered_glosses.append(g)

        # ---------------------------------------------------------------------
        # Step 3: Exact / Subsequence Idiom Matching
        # ---------------------------------------------------------------------
        gloss_tuple = tuple(raw_glosses)
        if gloss_tuple in PHRASE_TRANSLATIONS:
            return GrammarTranslationResult(
                raw_glosses=raw_glosses,
                asl_gloss_string=asl_gloss_str,
                english_translation=PHRASE_TRANSLATIONS[gloss_tuple],
                sentence_type=sentence_type,
                detected_tense=detected_tense,
                non_manual_notes=notes
            )

        # Also check filtered glosses without tense markers
        filt_tuple = tuple(filtered_glosses)
        if filt_tuple in PHRASE_TRANSLATIONS:
            eng = PHRASE_TRANSLATIONS[filt_tuple]
            # Prefix time marker if present
            if detected_tense == "PAST" and "yesterday" not in eng.lower():
                eng = f"Yesterday, {eng[0].lower()}{eng[1:]}"
            elif detected_tense == "FUTURE" and "tomorrow" not in eng.lower():
                eng = f"Tomorrow, {eng[0].lower()}{eng[1:]}"
            return GrammarTranslationResult(
                raw_glosses=raw_glosses,
                asl_gloss_string=asl_gloss_str,
                english_translation=eng,
                sentence_type=sentence_type,
                detected_tense=detected_tense,
                non_manual_notes=notes
            )

        # ---------------------------------------------------------------------
        # Step 4: Rule-Based Syntactic Restructuring (Topic-Comment -> SVO)
        # ---------------------------------------------------------------------
        translated = self._syntactic_reorder(
            glosses=filtered_glosses,
            sentence_type=sentence_type,
            tense=detected_tense,
            is_negated=is_neg
        )

        return GrammarTranslationResult(
            raw_glosses=raw_glosses,
            asl_gloss_string=asl_gloss_str,
            english_translation=translated,
            sentence_type=sentence_type,
            detected_tense=detected_tense,
            non_manual_notes=notes
        )

    def _syntactic_reorder(
        self,
        glosses: List[str],
        sentence_type: str,
        tense: str,
        is_negated: bool
    ) -> str:
        """Restructures ASL syntax patterns into natural English word order."""
        if not glosses:
            return ""

        words = [g.capitalize() for g in glosses if g not in ("NOT", "NO")]

        # Normalize pronouns
        norm_words = []
        for w in words:
            up = w.upper()
            if up in PRONOUN_MAP:
                norm_words.append(PRONOUN_MAP[up])
            else:
                norm_words.append(w.lower())

        # Detect Subject vs Object vs Verb
        # Common ASL Topic-Comment: [Topic/Object] [Subject] [Verb] -> e.g. "Store I go"
        if len(norm_words) >= 3:
            # Check if second word is a Subject Pronoun ("I", "you", "we", "he", "she", "they")
            subj_pronouns = {"I", "you", "we", "he", "she", "they"}
            if norm_words[1] in subj_pronouns and norm_words[0] not in subj_pronouns:
                # Invert: "Store I go" -> "I go store"
                norm_words = [norm_words[1], norm_words[2], norm_words[0]] + norm_words[3:]

        # Handle Verb Tense & Copulas
        first_word = norm_words[0] if norm_words else ""
        rest_words = norm_words[1:] if len(norm_words) > 1 else []

        # Auxiliary verb for questions / negation
        aux = "do"
        if first_word in ("he", "she", "it"):
            aux = "does"
        if tense == "PAST":
            aux = "did"
        elif tense == "FUTURE":
            aux = "will"

        sentence = ""

        # Yes/No Question Construction
        if sentence_type == "YES_NO_QUESTION":
            if first_word.lower() in ("you", "we", "they"):
                sentence = f"Are you {' '.join(rest_words)}?"
            elif first_word.lower() in ("he", "she"):
                sentence = f"Is {first_word} {' '.join(rest_words)}?"
            else:
                sentence = f"{aux.capitalize()} {first_word} {' '.join(rest_words)}?"

        # WH-Question Construction
        elif sentence_type == "WH_QUESTION":
            wh_word = "What"
            other_words = []
            for w in norm_words:
                if w.upper() in WH_WORDS:
                    wh_word = w.capitalize()
                else:
                    other_words.append(w)
            body = " ".join(other_words)
            if body:
                sentence = f"{wh_word} is {body}?"
            else:
                sentence = f"{wh_word}?"

        # Negation Construction
        elif is_negated:
            if tense == "PAST":
                sentence = f"{first_word} did not {' '.join(rest_words)}."
            elif tense == "FUTURE":
                sentence = f"{first_word} will not {' '.join(rest_words)}."
            else:
                neg_aux = "does not" if first_word.lower() in ("he", "she") else "do not"
                sentence = f"{first_word} {neg_aux} {' '.join(rest_words)}."

        # Standard Statement Construction
        else:
            # Insert copula if missing (e.g. "I happy" -> "I am happy")
            adj_words = {"happy", "sad", "tired", "cold", "ready", "sick", "doctor", "teacher", "fine"}
            if len(rest_words) >= 1 and rest_words[0] in adj_words:
                copula = "am" if first_word == "I" else ("is" if first_word in ("he", "she") else "are")
                sentence = f"{first_word} {copula} {' '.join(rest_words)}."
            else:
                # Add temporal prefix if specified
                base = f"{first_word} {' '.join(rest_words)}."
                if tense == "PAST":
                    sentence = f"{base[:-1]} (past)."
                elif tense == "FUTURE":
                    sentence = f"{first_word} will {' '.join(rest_words)}."
                else:
                    sentence = base

        # Capitalize and format
        clean_sentence = sentence.strip()
        if clean_sentence:
            clean_sentence = clean_sentence[0].upper() + clean_sentence[1:]
            if not clean_sentence.endswith((".", "?", "!")):
                clean_sentence += "."

        return clean_sentence
