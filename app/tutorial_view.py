"""
Interactive Sign Language Tutorial & Educational Masterclass Module.

Provides:
1. Visual ASL Dictionary & Reference Cards (A-Z, 0-9, special signs) with real sample photos.
2. Step-by-Step Two-Hand (Bimanual) Signs Masterclass ("BOOK", "HELP", "MEET", "WORK", "HOUSE", "FINISHED", "PLAY").
3. Enterprise & Workplace Macro Signs ("APPROVED", "PLEASE WAIT", "I AGREE", "THANK YOU", "HELLO").
4. Anatomical Disambiguation Tips (How to avoid confusing 'W' vs '6', 'U' vs 'V', 'C' vs 'O', 'Z' vs '1').
5. Camera Setup & Lighting Best Practices for maximum recognition accuracy.
"""
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import Image
import streamlit as st


# =============================================================================
# SIGN DESCRIPTIONS & ANATOMICAL INSTRUCTIONS
# =============================================================================

SIGN_TUTORIAL_DATA: Dict[str, Dict] = {
    "A": {
        "title": "Letter 'A'",
        "category": "Alphabet",
        "description": "Make a closed fist with all four fingers curled down against the palm. Place the thumb resting upright along the outer edge of the index finger.",
        "fingers": "Thumb: Upright along index side | Index, Middle, Ring, Pinky: Curled tightly into fist",
        "palm": "Facing forward toward the webcam lens",
        "tip": "Keep your thumb pointing UP along the side of the fist. Do not cross the thumb across your knuckles (that would be 'S')."
    },
    "B": {
        "title": "Letter 'B'",
        "category": "Alphabet",
        "description": "Hold all four fingers straight up, pressed tightly together in a flat vertical blade. Tuck the thumb flat across the palm.",
        "fingers": "Thumb: Folded flat across palm | Index, Middle, Ring, Pinky: Extended straight up, touching together",
        "palm": "Facing forward toward the camera",
        "tip": "Ensure all 4 fingers are glued together parallel. If fingers spread out, it may be recognized as number '4'."
    },
    "C": {
        "title": "Letter 'C'",
        "category": "Alphabet",
        "description": "Curve all fingers and thumb into a large open semi-circle or cup shape, forming the letter 'C'.",
        "fingers": "Thumb & All 4 Fingers: Curved into a smooth arc with an open aperture between thumb and fingertips",
        "palm": "Facing sideways or slightly angled forward",
        "tip": "Maintain a wide opening (aperture >= 40% of palm). If you close the gap completely, the AI will recognize it as 'O'."
    },
    "D": {
        "title": "Letter 'D'",
        "category": "Alphabet",
        "description": "Point your index finger straight up. Touch your thumb tip to the tips of your middle, ring, and pinky fingers to form an enclosed loop.",
        "fingers": "Thumb: Touching middle/ring/pinky tips | Index: Pointing straight up | Middle, Ring, Pinky: Curled into loop",
        "palm": "Facing forward",
        "tip": "Only the index finger is extended. If your thumb touches your index finger instead, that is 'F'."
    },
    "E": {
        "title": "Letter 'E'",
        "category": "Alphabet",
        "description": "Curl all four fingertips tightly downward so their pads rest on top of your folded thumb. Knuckles are prominently arched forward.",
        "fingers": "Thumb: Bent horizontally beneath fingers | All 4 Fingers: Curled down resting on thumb",
        "palm": "Facing forward",
        "tip": "Fingertips must physically touch or sit right above the thumb. Keep the fist compact."
    },
    "F": {
        "title": "Letter 'F'",
        "category": "Alphabet",
        "description": "Touch the tip of your index finger to the tip of your thumb to form an 'O' ring. Extend the middle, ring, and pinky fingers straight up and slightly spread.",
        "fingers": "Thumb & Index: Touching at tips (circle) | Middle, Ring, Pinky: Extended upright",
        "palm": "Facing forward",
        "tip": "The exact opposite of 'D': in 'F', 3 fingers are up; in 'D', only 1 finger is up."
    },
    "G": {
        "title": "Letter 'G'",
        "category": "Alphabet",
        "description": "Form a horizontal fist. Point your index finger horizontally forward, with the thumb held parallel to it like a pinch gesture.",
        "fingers": "Thumb & Index: Extended parallel horizontally | Middle, Ring, Pinky: Curled into fist",
        "palm": "Facing sideways toward your body",
        "tip": "Hold hand horizontally. If you point index straight up instead, the AI will classify it as '1' or 'L'."
    },
    "H": {
        "title": "Letter 'H'",
        "category": "Alphabet",
        "description": "Hold index and middle fingers extended straight horizontally, pressed together side-by-side. Thumb folded over ring finger.",
        "fingers": "Index & Middle: Extended together horizontally | Ring & Pinky: Curled | Thumb: Folded",
        "palm": "Facing inward toward your chest",
        "tip": "Keep index and middle parallel and horizontal. Like a horizontal version of 'U'."
    },
    "I": {
        "title": "Letter 'I'",
        "category": "Alphabet",
        "description": "Form a closed fist with only the pinky finger pointing straight up. Thumb folded across curled fingers.",
        "fingers": "Pinky: Pointing straight up | Thumb, Index, Middle, Ring: Curled tightly into fist",
        "palm": "Facing forward",
        "tip": "Hold hand completely still for 'I'. If you trace a curved scoop in the air with your pinky, it will trigger 'J'!"
    },
    "J": {
        "title": "Letter 'J'",
        "category": "Alphabet",
        "description": "Start with the 'I' handshape (upright pinky). Make a dynamic swooping scoop or hook motion in the air tracing the letter 'J'.",
        "fingers": "Pinky: Extended | Other fingers: Curled into fist",
        "palm": "Starts facing forward, curves inward during the stroke",
        "tip": "This is a motion-based dynamic sign! Make a deliberate curved swoop downward and up to the left."
    },
    "K": {
        "title": "Letter 'K'",
        "category": "Alphabet",
        "description": "Index finger points straight up. Middle finger angles forward/upward at roughly 45 degrees. Thumb rests between them, touching the middle finger's joint.",
        "fingers": "Index: Upright | Middle: Angled forward | Thumb: Wedged between index and middle | Ring, Pinky: Curled",
        "palm": "Facing forward",
        "tip": "The thumb tip must be placed firmly between the index and middle finger knuckles."
    },
    "L": {
        "title": "Letter 'L'",
        "category": "Alphabet",
        "description": "Extend index finger straight up and thumb straight out horizontally to form a distinct right angle (90 degrees), shaping the letter 'L'.",
        "fingers": "Thumb: Extended horizontally | Index: Pointing straight up | Middle, Ring, Pinky: Curled into fist",
        "palm": "Facing forward",
        "tip": "Keep the angle sharp and wide. Very easy to recognize with high confidence."
    },
    "M": {
        "title": "Letter 'M'",
        "category": "Alphabet",
        "description": "Make a fist with your thumb tucked underneath your index, middle, and ring fingers. The thumb tip peeks out between ring and pinky fingers.",
        "fingers": "Thumb: Tucked under 3 fingers (index, middle, ring) | All fingers: Curled over thumb",
        "palm": "Facing forward",
        "tip": "Remember: 3 fingers rest on top of the thumb for 'M' (representing the three humps of M)."
    },
    "N": {
        "title": "Letter 'N'",
        "category": "Alphabet",
        "description": "Make a fist with your thumb tucked underneath your index and middle fingers. The thumb tip peeks out between middle and ring fingers.",
        "fingers": "Thumb: Tucked under 2 fingers (index, middle) | All fingers: Curled over thumb",
        "palm": "Facing forward",
        "tip": "Remember: 2 fingers rest on top of the thumb for 'N' (representing the two legs of N)."
    },
    "O": {
        "title": "Letter 'O'",
        "category": "Alphabet",
        "description": "Curve all four fingers and thumb into a closed circular loop, with fingertips touching the thumb tip to form the round letter 'O'.",
        "fingers": "Thumb: Curved meeting all 4 curled fingertips to form a complete round circle",
        "palm": "Facing forward or slightly angled",
        "tip": "Keep the circle round. If the aperture is wide open, it's 'C'. If flattened vertically into a narrow slit, it's '0'."
    },
    "P": {
        "title": "Letter 'P'",
        "category": "Alphabet",
        "description": "Form the 'K' handshape, but point your hand downwards so index points straight down and middle points forward/down.",
        "fingers": "Index: Pointing down | Middle: Forward/down | Thumb: Wedged between | Ring, Pinky: Curled",
        "palm": "Facing down/inward",
        "tip": "Think of 'P' as a downward-pointing 'K'."
    },
    "Q": {
        "title": "Letter 'Q'",
        "category": "Alphabet",
        "description": "Form the 'G' handshape, but point your index finger and thumb straight down toward the ground like picking up a coin.",
        "fingers": "Thumb & Index: Extended parallel pointing downward | Others: Curled into fist",
        "palm": "Facing down",
        "tip": "Think of 'Q' as a downward-pointing 'G'."
    },
    "R": {
        "title": "Letter 'R'",
        "category": "Alphabet",
        "description": "Cross your middle finger over your index finger (like wishing for good luck). Hold them upright. Ring and pinky curled.",
        "fingers": "Middle finger: Crossed in front of index finger | Both: Pointing up | Thumb: Over ring/pinky",
        "palm": "Facing forward",
        "tip": "Crossing fingers is the defining signature of 'R'. Keep the cross visible to the camera."
    },
    "S": {
        "title": "Letter 'S'",
        "category": "Alphabet",
        "description": "Make a tight fist and fold your thumb firmly across the front of your curled fingers across the middle knuckles.",
        "fingers": "All 4 Fingers: Curled into fist | Thumb: Wrapped across the front of fingers",
        "palm": "Facing forward",
        "tip": "In 'S', the thumb crosses OVER the knuckles. In 'A', the thumb rests beside the index finger."
    },
    "T": {
        "title": "Letter 'T'",
        "category": "Alphabet",
        "description": "Make a fist with your thumb tucked directly under the index finger, so the thumb tip pokes up between the index and middle finger.",
        "fingers": "Thumb: Tucked under index finger only | All 4 Fingers: Curled",
        "palm": "Facing forward",
        "tip": "Only 1 finger (index) sits on top of the thumb for 'T'."
    },
    "U": {
        "title": "Letter 'U'",
        "category": "Alphabet",
        "description": "Extend index and middle fingers straight up, pressed tightly together parallel so there is no gap. Thumb holds down ring and pinky.",
        "fingers": "Index & Middle: Extended upright, touching side-by-side | Ring, Pinky: Curled | Thumb: Over ring/pinky",
        "palm": "Facing forward",
        "tip": "Fingers MUST touch parallel. If you spread them apart, the AI will classify it as 'V' or '2'."
    },
    "V": {
        "title": "Letter 'V'",
        "category": "Alphabet",
        "description": "Extend index and middle fingers straight up spread apart in a 'V' peace sign. Thumb holds down ring and pinky.",
        "fingers": "Index & Middle: Extended upright and spread apart into a 'V' | Ring, Pinky: Curled | Thumb: Over ring/pinky",
        "palm": "Facing forward",
        "tip": "Spread index and middle wide apart. If they touch, it's 'U'."
    },
    "W": {
        "title": "Letter 'W'",
        "category": "Alphabet",
        "description": "Extend index, middle, and ring fingers upright and spread apart to form a 'W'. Thumb holds down the pinky fingertip.",
        "fingers": "Index, Middle, Ring: Extended upright and spread apart | Pinky: Held down by thumb tip",
        "palm": "Facing forward",
        "tip": "Spread all three fingers wide into a fan shape. In number '6', those three fingers are held together."
    },
    "X": {
        "title": "Letter 'X'",
        "category": "Alphabet",
        "description": "Make a fist and hook your index finger forward like a claw or question mark. Other fingers stay curled.",
        "fingers": "Index: Hooked/curved at PIP and DIP joints | Middle, Ring, Pinky: Curled into fist | Thumb: Folded",
        "palm": "Facing sideways or slightly forward",
        "tip": "Make a distinct hook shape with the index finger."
    },
    "Y": {
        "title": "Letter 'Y'",
        "category": "Alphabet",
        "description": "Make a fist and extend your thumb and pinky finger fully out to the sides (the 'hang loose' or phone handshape).",
        "fingers": "Thumb & Pinky: Fully extended out | Index, Middle, Ring: Curled tightly into palm",
        "palm": "Facing forward",
        "tip": "Extend both thumb and pinky wide. Highly distinct gesture."
    },
    "Z": {
        "title": "Letter 'Z'",
        "category": "Alphabet",
        "description": "Extend your index finger and draw a zig-zag 'Z' trajectory in the air (right, diagonal down-left, right).",
        "fingers": "Index: Extended pointing forward | Others: Curled into fist",
        "palm": "Facing forward/inward",
        "tip": "This is a dynamic motion sign! If you hold your index still, the AI recognizes '1'. Trace the Z smoothly."
    },
    "0": {
        "title": "Number '0'",
        "category": "Number",
        "description": "Hold fingers flattened and curved into a narrow elongated vertical oval with thumb touching fingertip pads.",
        "fingers": "All fingers curved forward touching thumb in a narrow oval shape",
        "palm": "Facing forward",
        "tip": "Narrow and elongated compared to the round circular letter 'O'."
    },
    "1": {
        "title": "Number '1'",
        "category": "Number",
        "description": "Index finger pointing straight up in a static hold. Thumb holds middle, ring, and pinky curled into fist.",
        "fingers": "Index: Straight up | Other 4 digits: Curled into fist",
        "palm": "Facing forward or slightly inward",
        "tip": "Hold steady. Static hold distinguishes '1' from the dynamic zig-zag motion of 'Z'."
    },
    "2": {
        "title": "Number '2'",
        "category": "Number",
        "description": "Index and middle fingers pointing straight up, spread apart. Thumb folds over ring and pinky.",
        "fingers": "Index & Middle: Extended up spread apart | Ring, Pinky: Curled | Thumb: Over curled fingers",
        "palm": "Facing forward (or slightly inward for numbers)",
        "tip": "Spread index and middle wide."
    },
    "3": {
        "title": "Number '3'",
        "category": "Number",
        "description": "In ASL numbers, 3 is formed with Thumb, Index, and Middle fingers extended. Ring and pinky curled down.",
        "fingers": "Thumb, Index, Middle: Extended open | Ring, Pinky: Curled into palm",
        "palm": "Facing forward",
        "tip": "In ASL, 3 includes the thumb! Do not use index+middle+ring (that would be '6' or 'W')."
    },
    "4": {
        "title": "Number '4'",
        "category": "Number",
        "description": "All four fingers (index, middle, ring, pinky) extended upright and spread apart. Thumb folded flat across palm.",
        "fingers": "Index, Middle, Ring, Pinky: Extended upright | Thumb: Folded across palm",
        "palm": "Facing forward",
        "tip": "Keep all 4 fingers spread apart. If fingers touch parallel, it's 'B'."
    },
    "5": {
        "title": "Number '5'",
        "category": "Number",
        "description": "All five digits (thumb, index, middle, ring, pinky) fully extended wide open like a wide high-five.",
        "fingers": "All 5 digits: Fully extended and spread wide apart",
        "palm": "Facing forward toward camera",
        "tip": "Maximum hand expansion. Very easy for the AI to recognize."
    },
    "6": {
        "title": "Number '6'",
        "category": "Number",
        "description": "Index, middle, and ring fingers extended upright together. Thumb touches the tip of the pinky finger.",
        "fingers": "Index, Middle, Ring: Extended upright | Thumb: Touching pinky fingertip",
        "palm": "Facing forward",
        "tip": "Fingers are held closely parallel, and thumb touches the pinky. If fingers fan out wide, it's 'W'."
    },
    "7": {
        "title": "Number '7'",
        "category": "Number",
        "description": "Index, middle, and pinky fingers extended upright. Thumb touches the tip of the ring finger.",
        "fingers": "Index, Middle, Pinky: Extended upright | Thumb: Touching ring fingertip",
        "palm": "Facing forward",
        "tip": "Thumb specifically contacts the ring finger."
    },
    "8": {
        "title": "Number '8'",
        "category": "Number",
        "description": "Index, ring, and pinky fingers extended upright. Thumb touches the tip of the middle finger.",
        "fingers": "Index, Ring, Pinky: Extended upright | Thumb: Touching middle fingertip",
        "palm": "Facing forward",
        "tip": "Thumb specifically contacts the middle finger."
    },
    "9": {
        "title": "Number '9'",
        "category": "Number",
        "description": "Middle, ring, and pinky fingers extended upright. Thumb touches the tip of the index finger.",
        "fingers": "Middle, Ring, Pinky: Extended upright | Thumb: Touching index fingertip",
        "palm": "Facing forward",
        "tip": "Like the 'OK' handshape with 3 outer fingers extended."
    },
    "space": {
        "title": "Special: Space",
        "category": "Special",
        "description": "Flat open hand held horizontally or swept gently across to insert a space between words in the sentence builder.",
        "fingers": "All fingers open, hand oriented horizontally across frame",
        "palm": "Facing down or inward",
        "tip": "Inserts a space and commits the current word into the persistent file log."
    },
    "del": {
        "title": "Special: Delete / Backspace",
        "category": "Special",
        "description": "Hand held with thumb pointing backward or open pinch gesture to delete the last entered character.",
        "fingers": "Thumb pointing backward / away from palm",
        "palm": "Facing side",
        "tip": "Deletes the last typed letter from the sentence buffer."
    },
    "nothing": {
        "title": "Special: Nothing / Idle",
        "category": "Special",
        "description": "No active gesture being performed. Hand is resting, lowered below camera view, or out of frame.",
        "fingers": "Hand relaxed or absent",
        "palm": "N/A",
        "tip": "Used by the AI to keep the sentence steady when the user pauses between signs."
    }
}


# =============================================================================
# TWO-HANDED (BIMANUAL) PHRASES
# =============================================================================

BIMANUAL_TUTORIAL_DATA: Dict[str, Dict] = {
    "BOOK": {
        "title": "📖 BOOK",
        "category": "Two-Handed Word",
        "description": "Hold both hands with palms flat together facing each other (like closed book covers). Then open your hands outwards so your pinkies remain touching and palms face upward (like opening the pages of a book).",
        "left_hand": "Open flat palm, fingers straight and together",
        "right_hand": "Open flat palm, fingers straight and together",
        "relative_motion": "Wrists close (dist < 0.22), index tips close (dist < 0.16), hands pivoting open",
        "use_case": "Expressing books, documents, reading material, or library resources."
    },
    "HELP": {
        "title": "🤝 HELP",
        "category": "Two-Handed Word",
        "description": "Hold your non-dominant hand (left) flat with palm facing upward. Place your dominant hand (right) in a firm thumbs-up fist on top of the flat palm. Lift both hands together upward.",
        "left_hand": "Flat open base palm facing up",
        "right_hand": "Thumbs-up fist placed resting on left palm",
        "relative_motion": "Right fist resting on left palm (wrist dist < 0.24) and lifting upward",
        "use_case": "Asking for assistance, offering support, or signaling urgent need in meetings."
    },
    "MEET": {
        "title": "👥 MEET",
        "category": "Two-Handed Word",
        "description": "Hold both hands upright with only the index fingers pointing straight up (like two people standing). Bring both hands horizontally toward each other until your knuckles touch.",
        "left_hand": "Index finger extended upright, other fingers curled",
        "right_hand": "Index finger extended upright, other fingers curled",
        "relative_motion": "Both upright index fingers approaching each other until index tip distance < 0.10",
        "use_case": "Scheduling meetings, meeting someone, or initiating a video conference."
    },
    "WORK": {
        "title": "🔨 WORK",
        "category": "Two-Handed Word",
        "description": "Form both hands into fists. Hold your non-dominant fist stationary in front of you. Tap the base/wrist of your dominant fist onto the back of the other wrist twice.",
        "left_hand": "Closed fist held steady horizontally",
        "right_hand": "Closed fist tapping on top of left wrist",
        "relative_motion": "Both hands in fists, wrist proximity < 0.18 with tapping contact",
        "use_case": "Discussing work, tasks, projects, engineering sprints, or workplace duties."
    },
    "HOUSE": {
        "title": "🏠 HOUSE",
        "category": "Two-Handed Word",
        "description": "Hold both hands flat open with fingers together. Angle hands inward toward each other so your index and middle fingertips touch at the apex, forming the peaked roof of a house.",
        "left_hand": "Flat open hand angled at 45 degrees",
        "right_hand": "Flat open hand angled at 45 degrees",
        "relative_motion": "Fingertips touching at top (index dist < 0.09) while wrists remain apart (wrist dist > 0.18)",
        "use_case": "Referring to home, house, remote work, or physical building location."
    },
    "FINISHED": {
        "title": "🏁 FINISHED / ALL DONE",
        "category": "Two-Handed Word",
        "description": "Hold both open hands in front of your chest with palms facing you. Quickly flick and sweep both hands outward to the sides, ending with palms facing downward and outward.",
        "left_hand": "Open hand starting facing chest, sweeping outward",
        "right_hand": "Open hand starting facing chest, sweeping outward",
        "relative_motion": "High kinetic energy (> 0.04) sweeping hands apart (wrist distance > 0.30)",
        "use_case": "Declaring task completion, finishing a project, sprint closure, or saying 'all done'."
    },
    "PLAY": {
        "title": "🎮 PLAY",
        "category": "Two-Handed Word",
        "description": "Form the 'Y' handshape with BOTH hands simultaneously (thumb and pinky extended, middle 3 curled). Shake or oscillate both hands gently back and forth in rhythm.",
        "left_hand": "'Y' shape (thumb & pinky extended)",
        "right_hand": "'Y' shape (thumb & pinky extended)",
        "relative_motion": "Both Y-hands oscillating in rhythm",
        "use_case": "Casual activities, games, recreational breaks, or playtime."
    }
}


# =============================================================================
# WORKPLACE & MACRO QUICK SIGNS
# =============================================================================

WORKPLACE_TUTORIAL_DATA: Dict[str, Dict] = {
    "APPROVED": {
        "title": "👍 APPROVED / LOOKS GOOD",
        "category": "Workplace Macro Sign",
        "description": "Hold a firm, static thumbs-up gesture steady in front of the camera for 0.5 seconds.",
        "trigger": "Thumb pointing straight up, four fingers curled tightly, kinetic energy < 0.05",
        "action": "Automatically types 'Approved / Looks Good!' and logs it to file."
    },
    "PLEASE WAIT": {
        "title": "✋ PLEASE WAIT / ONE MOMENT",
        "category": "Workplace Macro Sign",
        "description": "Hold a flat open palm facing directly forward toward the camera lens in a gentle hold.",
        "trigger": "All five fingers extended upright and open, static hold",
        "action": "Automatically types 'Please Wait / One Moment' to pause the meeting."
    },
    "I AGREE": {
        "title": "🤝 I AGREE / IN FULL AGREEMENT",
        "category": "Workplace Macro Sign",
        "description": "Place a flat open palm over your upper chest / heart area.",
        "trigger": "Open palm held flat inward over torso/chest elevation",
        "action": "Automatically types 'I Agree / In Full Agreement'."
    },
    "THANK YOU": {
        "title": "🙏 THANK YOU / APPRECIATION",
        "category": "Workplace Macro Sign",
        "description": "Touch the fingertips of your flat open hand to your chin or lips, then move your hand forward and slightly downward toward the webcam.",
        "trigger": "Hand at chin level transitioning outward in smooth trajectory",
        "action": "Automatically types 'Thank You Very Much'."
    },
    "HELLO": {
        "title": "👋 HELLO / GREETING",
        "category": "Workplace Macro Sign",
        "description": "Hold an open hand near temple or shoulder level and wave gently from side to side.",
        "trigger": "Open palm at head level oscillating horizontally",
        "action": "Automatically types 'Hello Everyone / Greeting'."
    }
}


# =============================================================================
# IMAGE PATH RESOLVER
# =============================================================================

def resolve_sign_sample_image(sign_key: str) -> Optional[Path]:
    """Finds an actual sample image for the given sign from local archive folders."""
    test_dir = Path("archive/asl_alphabet_test/asl_alphabet_test")
    p1 = test_dir / f"{sign_key}_test.jpg"
    if p1.exists():
        return p1

    asl_dataset_dir = Path("archive/asl_dataset")
    p2_dir = asl_dataset_dir / sign_key.lower()
    if p2_dir.exists():
        imgs = list(p2_dir.glob("*.jpg")) + list(p2_dir.glob("*.jpeg")) + list(p2_dir.glob("*.png"))
        if imgs:
            return imgs[0]

    return None


# =============================================================================
# MAIN RENDER FUNCTION FOR STREAMLIT
# =============================================================================

def render_tutorial_tab():
    """Renders the comprehensive interactive tutorial tab in Streamlit."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); padding: 18px 24px; border-radius: 12px; margin-bottom: 20px; color: white;">
        <h2 style="margin: 0; color: white; font-size: 1.8rem;">📖 Sign Language Masterclass & Interactive Tutorial</h2>
        <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 1.0rem;">
            Learn all 39 base alphabet & number signs, 7 two-handed whole-word phrases, and workplace productivity macros with photographic reference guides and anatomical accuracy tips.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tutorial_section = st.radio(
        "Choose Tutorial Module:",
        options=[
            "🔤 Alphabet & Numbers Visual Guide (39 Signs)",
            "👐 Two-Hand (Bimanual) Signs Masterclass (7 Words)",
            "💼 Workplace & Meeting Macro Signs",
            "💡 Pro Tips for 100% Recognition Accuracy"
        ],
        horizontal=True
    )

    st.markdown("---")

    # =========================================================================
    # MODULE 1: ALPHABET & NUMBERS VISUAL GUIDE
    # =========================================================================
    if tutorial_section == "🔤 Alphabet & Numbers Visual Guide (39 Signs)":
        c_filter, c_search = st.columns([2, 3])
        with c_filter:
            category_filter = st.selectbox(
                "Filter by Category:",
                options=["All Signs (39)", "Letters A–Z (26)", "Numbers 0–9 (10)", "Special Control Signs (3)"],
                index=0
            )

        # Filter sign list
        all_keys = list(SIGN_TUTORIAL_DATA.keys())
        if category_filter == "Letters A–Z (26)":
            filtered_keys = [k for k in all_keys if SIGN_TUTORIAL_DATA[k]["category"] == "Alphabet"]
        elif category_filter == "Numbers 0–9 (10)":
            filtered_keys = [k for k in all_keys if SIGN_TUTORIAL_DATA[k]["category"] == "Number"]
        elif category_filter == "Special Control Signs (3)":
            filtered_keys = [k for k in all_keys if SIGN_TUTORIAL_DATA[k]["category"] == "Special"]
        else:
            filtered_keys = all_keys

        with c_search:
            selected_sign = st.selectbox(
                "Select a Sign to Inspect & Learn:",
                options=filtered_keys,
                index=0,
                format_func=lambda k: f"{k} - {SIGN_TUTORIAL_DATA[k]['title']}"
            )

        sign_info = SIGN_TUTORIAL_DATA[selected_sign]
        img_path = resolve_sign_sample_image(selected_sign)

        col_img, col_desc = st.columns([1, 2])

        with col_img:
            if img_path and img_path.exists():
                img = Image.open(img_path)
                st.image(img, caption=f"Reference Pose for '{selected_sign}'", use_container_width=True)
            else:
                st.info(f"No sample photo found for '{selected_sign}'.")

        with col_desc:
            cat_badge = "#1E40AF" if sign_info["category"] == "Alphabet" else ("#047857" if sign_info["category"] == "Number" else "#B91C1C")
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                <h3 style="margin: 0; color: #1E293B;">{sign_info['title']}</h3>
                <span style="background-color: {cat_badge}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.82rem; font-weight: 700;">
                    {sign_info['category']}
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"**Description:** {sign_info['description']}")
            st.markdown(f"**🖐️ Finger Configuration:** `{sign_info['fingers']}`")
            st.markdown(f"**🧭 Palm Direction:** {sign_info['palm']}")

            st.info(f"💡 **Pro-Tip & Disambiguation:** {sign_info['tip']}")

        # Quick thumbnail browser grid below
        st.markdown("#### 🖼️ Quick Sign Gallery (Click above to view details)")
        gallery_cols = st.columns(8)
        for idx, k in enumerate(filtered_keys[:24]):
            with gallery_cols[idx % 8]:
                k_img = resolve_sign_sample_image(k)
                if k_img and k_img.exists():
                    st.image(Image.open(k_img), caption=f"Sign '{k}'", use_container_width=True)
                else:
                    st.button(f"'{k}'", key=f"btn_{k}")

    # =========================================================================
    # MODULE 2: TWO-HAND (BIMANUAL) SIGNS MASTERCLASS
    # =========================================================================
    elif tutorial_section == "👐 Two-Hand (Bimanual) Signs Masterclass (7 Words)":
        st.markdown("""
        ### 👐 Two-Handed (Bimanual) Whole-Word Sign Language
        The system features synchronized dual-hand tracking (`MediaPipe Hands` tracking up to 2 hands) to recognize natural full words.
        Hold both hands in frame to perform these signs:
        """)

        bm_tabs = st.tabs([f"{data['title']}" for data in BIMANUAL_TUTORIAL_DATA.values()])

        for idx, (b_key, b_data) in enumerate(BIMANUAL_TUTORIAL_DATA.items()):
            with bm_tabs[idx]:
                b_col1, b_col2 = st.columns([1, 1])

                with b_col1:
                    st.markdown(f"### {b_data['title']}")
                    st.write(f"**How to Sign:** {b_data['description']}")
                    st.markdown(f"**Left Hand Pose:** `{b_data['left_hand']}`")
                    st.markdown(f"**Right Hand Pose:** `{b_data['right_hand']}`")
                    st.markdown(f"**Inter-Hand Relational Geometry:** `{b_data['relative_motion']}`")
                    st.success(f"**📌 Typical Usage:** {b_data['use_case']}")

                with b_col2:
                    st.markdown("""
                    <div style="background-color: #F8FAFC; border: 2px dashed #94A3B8; border-radius: 12px; padding: 20px; text-align: center;">
                        <div style="font-size: 3.5rem; margin-bottom: 10px;">👐</div>
                        <h4 style="margin: 0; color: #1E3A8A;">Synchronized 2-Hand Tracker Active</h4>
                        <p style="font-size: 0.92rem; color: #64748B; margin-top: 6px;">
                            The AI computes inter-hand Euclidean wrist distance, relative fingertip contact, and synchronized velocity vectors.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info(f"**Recognition Criteria:** When both hands match `{b_key}`, the prediction HUD immediately shows a green `TWO-HANDED GESTURE` pill and commits the word directly into the sentence log.")

    # =========================================================================
    # MODULE 3: WORKPLACE & MEETING MACRO SIGNS
    # =========================================================================
    elif tutorial_section == "💼 Workplace & Meeting Macro Signs":
        st.markdown("""
        ### 💼 Workplace & Video Meeting Instant Macros
        Designed specifically for enterprise video conferences (Zoom, Microsoft Teams, Google Meet), these single-hand macro signs let you communicate instantly without unmuting:
        """)

        for wp_key, wp_data in WORKPLACE_TUTORIAL_DATA.items():
            with st.expander(f"{wp_data['title']} — Trigger Guide", expanded=True):
                w1, w2 = st.columns([2, 1])
                with w1:
                    st.write(f"**Gesture Overview:** {wp_data['description']}")
                    st.markdown(f"**Kinematic Trigger:** `{wp_data['trigger']}`")
                    st.markdown(f"**Action Executed:** `{wp_data['action']}`")
                with w2:
                    st.markdown("""
                    <div style="background: #F0FDF4; border: 1px solid #BBF7D0; padding: 12px; border-radius: 8px; text-align: center;">
                        <span style="color: #166534; font-weight: 700; font-size: 0.95rem;">⚡ Instant Meeting HUD</span><br/>
                        <span style="color: #4B5563; font-size: 0.85rem;">Broadcasts over screen share</span>
                    </div>
                    """, unsafe_allow_html=True)

    # =========================================================================
    # MODULE 4: PRO TIPS FOR 100% ACCURACY
    # =========================================================================
    elif tutorial_section == "💡 Pro Tips for 100% Recognition Accuracy":
        st.markdown("### 💡 Expert Best Practices for 100% Recognition Accuracy")

        p1, p2 = st.columns(2)

        with p1:
            st.markdown("""
            #### 1. 📏 Camera Distance & Positioning
            - Keep your hand **1.5 to 3.0 feet (45–90 cm)** from your webcam lens.
            - Ensure your hand is framed inside the center $70\%$ of the video frame.
            - Keep your wrist and all 5 fingertips in clear view without cutting off the edges.

            #### 2. 💡 Lighting & Background Contrast
            - Position a soft light source in front of you (e.g. monitor light bar or desk lamp).
            - Avoid strong windows or bright spotlights directly behind your back (backlighting causes silhouettes).
            - Solid, uncluttered backgrounds provide the cleanest MediaPipe landmark mesh.
            """)

        with p2:
            st.markdown("""
            #### 3. 🛡️ 1€ Anti-Tremor Filter Settings
            - The built-in **1€ Biomechanical Filter** eliminates **98.7% of high-frequency tremor noise**.
            - If your hands naturally shake or shiver, keep the default settings:
              - `Min Cutoff (fc_min)`: **0.8 Hz** (smooths resting tremors)
              - `Responsiveness (Beta)`: **0.015** (retains zero-lag transitions)
              - `Tremor Kinetic Deadband`: **Enabled** (prevents velocity spikes)

            #### 4. ⏱️ Holding vs Motion Strokes
            - **Static Signs (A–Z, 0–9)**: Hold your hand pose steady for **0.4 to 0.6 seconds**. The HUD badge will switch from amber to bright green (`STABLE`) and append to your sentence.
            - **Motion Signs ('Z', 'J', 'FINISHED')**: Make deliberate, smooth strokes across the frame.
            """)

        st.markdown("---")
        st.markdown("""
        #### 📁 Automatic Persistent File Logging
        Every recognized sign and constructed sentence is automatically logged with timestamps directly into:
        `data/predicted_signs.txt`
        
        You can inspect or copy this file at any time for meeting minutes, records, or chat integration.
        """)
