"""
Text-to-ASL Synthesizer & 3D Keyframe Trajectory Generator
===========================================================
Converts spoken or typed natural English text into:
1. Grammatically structured ASL gloss sequences.
2. Timed 3D hand skeletal animation keyframes (21 landmarks x [x, y, z]).
3. Smooth transition trajectories between consecutive signs and rest postures.
"""

import re
import json
import sys
import os
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

# Ensure project root is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.canonical_sign_library import get_canonical_library, HAND_CONNECTIONS, FINGER_GROUPS


# Conversational Idioms & Direct Sign Mappings
IDIOM_MAP: Dict[str, List[str]] = {
    "HOW ARE YOU": ["HOW", "YOU"],
    "NICE TO MEET YOU": ["NICE", "MEET", "YOU"],
    "THANK YOU": ["THANK", "YOU"],
    "THANKS": ["THANK", "YOU"],
    "PLEASE HELP": ["PLEASE", "HELP"],
    "GOOD MORNING": ["GOOD", "MORNING"],
    "GOOD AFTERNOON": ["GOOD", "AFTERNOON"],
    "WHAT IS YOUR NAME": ["YOU", "NAME", "WHAT"],
    "WHERE IS THE BATHROOM": ["BATHROOM", "WHERE"],
    "I AGREE": ["I", "AGREE"],
    "I DO NOT UNDERSTAND": ["I", "UNDERSTAND", "NOT"]
}

CONTRACTION_MAP: Dict[str, str] = {
    "what's": "what is",
    "where's": "where is",
    "how's": "how is",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "can't": "can not",
    "don't": "do not",
    "didn't": "did not",
    "won't": "will not",
    "we're": "we are",
    "they're": "they are",
    "it's": "it is",
    "let's": "let us"
}

SIGN_DESCRIPTIONS: Dict[str, str] = {
    "A": "Fist with thumb resting alongside the index finger.",
    "B": "Flat hand pointing upward with thumb tucked across the palm.",
    "C": "Curved hand forming a C-shape with all fingers and thumb.",
    "D": "Index finger extended straight up; other fingers and thumb form a loop.",
    "E": "Fingers curled tightly with fingertips resting on the thumb.",
    "F": "Index finger and thumb touch in a circle; remaining 3 fingers spread upward.",
    "G": "Index finger and thumb extend horizontally parallel.",
    "H": "Index and middle fingers extend horizontally parallel.",
    "I": "Pinky finger extended upright; other fingers closed into a fist.",
    "J": "Pinky finger traces a J curve in the air.",
    "K": "Index and middle fingers point upward with thumb placed between them.",
    "L": "Index finger and thumb form an L-shape at a right angle.",
    "M": "Thumb tucked under the first three curled fingers.",
    "N": "Thumb tucked under the first two curled fingers.",
    "O": "All fingertips touch the thumb forming an O circle.",
    "P": "Similar to K, but oriented pointing downward.",
    "Q": "Index and thumb pointing downward like a downward G.",
    "R": "Index and middle fingers crossed tightly.",
    "S": "Fist with thumb wrapped across the front of all fingers.",
    "T": "Thumb tucked under the index finger only.",
    "U": "Index and middle fingers held together pointing upward.",
    "V": "Index and middle fingers spread apart forming a V-shape.",
    "W": "Index, middle, and ring fingers spread upward forming a W.",
    "X": "Index finger bent into a hook shape.",
    "Y": "Thumb and pinky finger extended outwards in a phone shape.",
    "Z": "Index finger draws a Z pattern in the air.",
    "SPACE": "Open hand held horizontally to separate words.",
    "REST": "Relaxed natural neutral hand posture.",
    "HELP": "Closed fist on flat palm moving upward.",
    "COMPUTER": "Curved C-hand articulating on forearm.",
    "CALL": "Hand near ear simulating a call.",
    "GO": "Both index fingers pointing forward and outward.",
    "BEFORE": "Flat hand moving backward toward the shoulder."
}


class TextToASLSynthesizer:
    """
    Transforms English text into timed 3D ASL animation keyframe sequences.
    """

    def __init__(self):
        self.library = get_canonical_library()
        self.rest_pose = self.library.get("REST", [[0.0, 0.0, 0.0]] * 21)

    def normalize_english(self, text: str) -> str:
        """Expands contractions and normalizes whitespace."""
        clean = text.lower().strip()
        for cont, full in CONTRACTION_MAP.items():
            clean = re.sub(r'\b' + re.escape(cont) + r'\b', full, clean)
        # Remove special characters except alphanumeric and spaces
        clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean.upper()

    def translate_to_asl_glosses(self, text: str) -> List[Dict[str, Any]]:
        """
        Decomposes input text into an ordered list of ASL sign tokens
        (word signs, fingerspelled characters, and word spacing).
        """
        normalized = self.normalize_english(text)
        if not normalized:
            return []

        tokens: List[Dict[str, Any]] = []

        # Check for multi-word idiom matches first
        matched_idiom = None
        for idiom, glosses in IDIOM_MAP.items():
            if idiom in normalized:
                # Replace first occurrence
                normalized = normalized.replace(idiom, " ".join(glosses))

        words = normalized.split()

        for w_idx, word in enumerate(words):
            # 1. Check if the whole word is directly in canonical library
            if word in self.library and word not in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
                tokens.append({
                    "token": word,
                    "type": "word",
                    "display": word,
                    "desc": SIGN_DESCRIPTIONS.get(word, f"Whole-word sign for '{word}'.")
                })
            else:
                # 2. Fingerspell the word letter by letter
                for ch in word:
                    if ch in self.library:
                        tokens.append({
                            "token": ch,
                            "type": "letter",
                            "display": ch,
                            "desc": SIGN_DESCRIPTIONS.get(ch, f"Fingerspelled letter '{ch}'.")
                        })

            # Add space between words
            if w_idx < len(words) - 1:
                tokens.append({
                    "token": "SPACE",
                    "type": "control",
                    "display": "␣",
                    "desc": "Word boundary pause and space."
                })

        return tokens

    def generate_animation_payload(
        self,
        text: str,
        hold_time_ms: int = 500,
        transition_time_ms: int = 250
    ) -> Dict[str, Any]:
        """
        Produces a complete JSON-ready animation payload containing:
        - 3D keyframe sequence with coordinates
        - Timeline synchronization data
        - Hand skeletal connections for rendering
        """
        gloss_tokens = self.translate_to_asl_glosses(text)
        if not gloss_tokens:
            return {
                "input_text": text,
                "tokens": [],
                "keyframes": [],
                "total_duration_ms": 0,
                "connections": HAND_CONNECTIONS,
                "finger_groups": FINGER_GROUPS
            }

        keyframes: List[Dict[str, Any]] = []
        current_time_ms = 0

        # Start from relaxed resting posture
        keyframes.append({
            "step_index": 0,
            "sign": "REST",
            "display": "Rest",
            "type": "posture",
            "landmarks": self.rest_pose,
            "start_time_ms": current_time_ms,
            "hold_time_ms": 200,
            "transition_time_ms": transition_time_ms,
            "desc": "Initial relaxed resting posture."
        })
        current_time_ms += 200 + transition_time_ms

        for idx, tok in enumerate(gloss_tokens):
            sign_key = tok["token"]
            pts = self.library.get(sign_key, self.rest_pose)

            # Control signs (like space) can have slightly shorter hold time
            h_time = 350 if tok["type"] == "control" else hold_time_ms

            keyframes.append({
                "step_index": idx + 1,
                "sign": sign_key,
                "display": tok["display"],
                "type": tok["type"],
                "landmarks": pts,
                "start_time_ms": current_time_ms,
                "hold_time_ms": h_time,
                "transition_time_ms": transition_time_ms,
                "desc": tok["desc"]
            })
            current_time_ms += h_time + transition_time_ms

        # Return to resting posture at conclusion
        keyframes.append({
            "step_index": len(gloss_tokens) + 1,
            "sign": "REST",
            "display": "Rest",
            "type": "posture",
            "landmarks": self.rest_pose,
            "start_time_ms": current_time_ms,
            "hold_time_ms": 300,
            "transition_time_ms": transition_time_ms,
            "desc": "Final neutral resting posture."
        })
        current_time_ms += 300

        return {
            "input_text": text,
            "tokens": gloss_tokens,
            "keyframes": keyframes,
            "total_duration_ms": current_time_ms,
            "connections": HAND_CONNECTIONS,
            "finger_groups": FINGER_GROUPS
        }


if __name__ == "__main__":
    synthesizer = TextToASLSynthesizer()
    payload = synthesizer.generate_animation_payload("Hello! Need help?")
    print(f"Generated {len(payload['keyframes'])} keyframes for '{payload['input_text']}'.")
    print(f"Total duration: {payload['total_duration_ms']} ms")
    for k in payload['keyframes'][:6]:
        print(f"  [{k['start_time_ms']} ms] Sign: {k['sign']} ({k['type']})")
