# 🤟 ASL AI Complete Feature & User Operating Guide
> **The Definitive Handbook for Real-Time Motion-Aware Sign Language Detection, Two-Way 3D Avatar Translation & Air-Workspace AI**

---

## 📑 Table of Contents
1. [Quickstart & Launching the System](#1-quickstart--launching-the-system)
2. [Hardware & Camera Setup (Zero-Lag & Multi-Backend)](#2-hardware--camera-setup-zero-lag--multi-backend)
3. [Tab 1: Live Motion Webcam & Dual Output](#3-tab-1-live-motion-webcam--dual-output)
   * [Dual Recognition Paradigms (Continuous vs Isolated)](#dual-recognition-paradigms)
   * [Dual Real-Time Output & Linguistic Grammar Badges](#dual-real-time-output--linguistic-grammar-badges)
   * [Transcript Building & Editing Controls](#transcript-building--editing-controls)
   * [Autocorrect, Autospace & Speech Synthesis](#autocorrect-autospace--speech-synthesis)
4. [The 5 Breakthrough AI Innovations](#4-the-5-breakthrough-ai-innovations)
   * [Innovation 1: Rapid Co-Articulation Kinematic Velocity Decoder](#innovation-1-rapid-co-articulation-kinematic-velocity-decoder)
   * [Innovation 2: Lip-Morpheme Facial Fusion Engine](#innovation-2-lip-morpheme-facial-fusion-engine)
   * [Innovation 3: Ambient Peripheral Vision HUD](#innovation-3-ambient-peripheral-vision-hud)
   * [Innovation 4: One-Shot Biomechanical Hand Calibration](#innovation-4-one-shot-biomechanical-hand-calibration)
   * [Innovation 5: Air-Workspace In-Meeting Gesture Macros](#innovation-5-air-workspace-in-meeting-gesture-macros)
5. [Tab 2: Reverse Speech-to-Sign 3D Avatar (Two-Way Communication)](#5-tab-2-reverse-speech-to-sign-3d-avatar-two-way-communication)
6. [Tab 3: Static Image Analysis & Dataset Explorer](#6-tab-3-static-image-analysis--dataset-explorer)
7. [Tab 4: Model Performance & Comprehensive Benchmarks](#7-tab-4-model-performance--comprehensive-benchmarks)
8. [Tab 5: Interactive Sign Tutorial & Practice Guide](#8-tab-5-interactive-sign-tutorial--practice-guide)
9. [Personal Vocabulary & Custom Terminology Manager](#9-personal-vocabulary--custom-terminology-manager)
10. [Accessibility & Theme Customization](#10-accessibility--theme-customization)
11. [Troubleshooting & FAQ](#11-troubleshooting--faq)

---

## 1. Quickstart & Launching the System

### Prerequisites
* Windows 10/11, macOS, or Linux.
* Python 3.10+ installed.
* Standard integrated webcam or external USB camera.

### Step 1: Install Dependencies
Open your terminal in the project directory (`d:\InnovativeProject`) and run:
```bash
python -m pip install -r requirements.txt
```

### Step 2: Launch Application
```bash
streamlit run app/app.py
```
The web dashboard will automatically open in your default browser at:
`http://localhost:8501`

---

## 2. Hardware & Camera Setup (Zero-Lag & Multi-Backend)

Located in the **Sidebar** under **🎥 Camera & Performance**:

| Control | Description & Best Practice |
|---|---|
| **Webcam Index** | Defaults to `0`. If your screen is pitch-black or your laptop has **Windows Hello Infrared (IR)**, switch this to **`1`** (the standard RGB camera). |
| **Capture Backend** | * **`Auto`** *(Recommended)*: Automatically chooses the fastest hardware backend.<br>* **`DirectShow`**: Instant connection ($<50\text{ms}$) on Windows with hardware MJPG compression (eliminates MSMF frame-rate starvation).<br>* **`Default`**: Generic OS-level capture.<br>* **`MSMF`**: Windows Media Foundation backend. |
| **FPS Mode** | * **🚀 Turbo (60 FPS Unlocked)**: High-speed SIMD JPEG encoding with zero frame buffering.<br>* **⚡ Balanced (30 FPS Standard)**: Optimized for laptops on battery.<br>* **🔋 Power Saver (20 FPS)**: Lowers CPU utilization. |
| **Auto-Switch Camera** | When checked, if index 0 is non-responsive or an unilluminated IR camera, the system automatically scans and promotes index 1 to active. |
| **🔍 Scan Cameras** | Probes your physical hardware and displays an active diagnostics table of every connected webcam with its native resolution. |
| **🔄 Force Release** | Instantly closes all OpenCV camera handles and destroys system locks if a device gets hung. |

> [!TIP]
> **Laptop Physical Privacy Slider**: Many Lenovo, HP, and Dell laptops have a mechanical slider switch directly above the webcam lens. If the camera stream is black, check that this mechanical slider is pushed to the open position.

---

## 3. Tab 1: Live Motion Webcam & Dual Output

### Dual Recognition Paradigms

You can toggle between two recognition paradigms in the sidebar:

#### Paradigm A: Isolated Motion-Aware Engine (LSTM + Random Forest)
* **Best For**: High-precision fingerspelling of names, alphanumeric codes, technical words, and individual signs.
* **Accuracy**: **99.3%** ensemble precision across 39 classes (`A-Z`, `0-9`, `space`, `del`, `nothing`).
* **1€ (One-Euro) Anti-Shiver Filter**: Dampens hand tremors, muscle vibrations, and camera jitter by **98.7%** without adding perceptual latency.
* **Whole-Word Signs Toggle**: In the sidebar, check *"Enable Whole-Word Signs (WLASL & Phrases)"* to enable 2-handed bimanual signs (`BOOK`, `HELP`, `MEET`, `WORK`, `HOUSE`, `FINISHED`, `PLAY`).

#### Paradigm B: Continuous ASL Stream (CNN-BiLSTM-CTC)
* **Best For**: Natural continuous signing streams without requiring signers to hold pauses between signs.
* **Holistic Multimodal Fusion**: Tracks hands, body pose, and facial non-manual markers simultaneously.
* **ASL-to-English Grammar Engine**: Translates ASL topic-comment word order into grammatically fluent English sentences.

---

### Dual Real-Time Output & Linguistic Grammar Badges

As you sign, the system generates two synchronized text channels:
1. **Raw ASL Glosses (Sign Order)**: Direct representation of gestures in the order they were produced (e.g. `STORE I GO`).
2. **Translated English (Grammar Aware)**: English sentence with proper articles, prepositions, and verb tenses (e.g. `I am going to the store.`).

#### Linguistic Grammar Badges
The system automatically classifies your facial markers (eyebrow raise, furrow, head tilt) into linguistic sentence categories:
* <span style="background:#FEF3C7; color:#92400E; padding:2px 8px; border-radius:6px; font-weight:bold;">🔍 WH-Question</span>: Triggered by furrowed eyebrows and head tilt (e.g., `NAME YOU WHAT` $\rightarrow$ *"What is your name?"*).
* <span style="background:#E0F2FE; color:#075985; padding:2px 8px; border-radius:6px; font-weight:bold;">❓ Yes/No Question</span>: Triggered by raised eyebrows and forward head tilt (e.g., `YOU WANT COFFEE` $\rightarrow$ *"Would you like coffee?"*).
* <span style="background:#FEE2E2; color:#991B1B; padding:2px 8px; border-radius:6px; font-weight:bold;">🚫 Negated Statement</span>: Triggered by side-to-side head shakes (e.g., `I UNDERSTAND` $\rightarrow$ *"I do not understand."*).
* <span style="background:#ECFDF5; color:#065F46; padding:2px 8px; border-radius:6px; font-weight:bold;">🗣️ Declarative</span>: Standard affirmative sentence structure.

---

### Transcript Building & Editing Controls

Located directly below the video feed and output boxes:

| Control | Action | Keyboard / Gesture Equivalent |
|---|---|---|
| **➕ Add Current** | Commits the currently predicted sign or letter into the active sentence. | Hold hand stable for 7 frames. |
| **␣ Space** | Commits the current word, applies spelling autocorrect, and inserts a space delimiter. | Horizontal flat palm gesture or 1.2s pause. |
| **⌫ Backspace / Undo Sign** | Deletes the last character or undoes the most recent continuous sign gloss. | Fist `del` gesture or Backspace button. |
| **🗑️ Delete Word** | Deletes the entire active word currently being fingerspelled. | UI button click. |
| **🧹 Clear All** | Clears the active sentence and resets the sequence buffer. | Sidebar or UI button. |
| **💾 Save Sentence Now** | Commits the finished sentence to the meeting transcript log file with a timestamp. | UI button click. |
| **🔊 Speak English (TTS)** | Speaks the translated English sentence out loud using offline neural text-to-speech. | UI button click. |
| **Top-3 Alternatives** | Interactive buttons (`#1`, `#2`, `#3`) showing alternative candidate predictions. Clicking any candidate immediately corrects the transcript and calibrates the personalization engine. | User click. |

---

## 4. The 5 Breakthrough AI Innovations

### Innovation 1: Rapid Co-Articulation Kinematic Velocity Decoder
* **Purpose**: Solves the biggest limitation of traditional sign language AI—the need for signers to artificially pause between letters.
* **How to Enable**: In the sidebar under **💡 AI Innovations**, toggle **`🚀 Rapid Co-Articulation Mode (5x Speed)`**.
* **How to Use**: Fingerspell words continuously at normal human speeds (4 to 6 letters per second). The engine computes instantaneous joint kinetic energy and detects the exact **inflection point (apex)** of tendon deceleration, committing each letter without requiring stationary holds.

---

### Innovation 2: Lip-Morpheme Facial Fusion Engine
* **Purpose**: Distinguishes visually identical or ambiguous hand gestures using subtle, unconscious mouthing syllables observed in native deaf signers.
* **How it Works**: Uses MediaPipe FaceMesh lip contours (landmarks 13, 14, 61, 291) to compute Mouth Aspect Ratio (MAR):
  * **`M` vs `N`**:
    * Say / mouth *"M"* $\rightarrow$ **Bilabial Closure** (lips pressed tightly, $\text{MAR} < 0.055$) $\rightarrow$ Emits **`M`**.
    * Say / mouth *"N"* $\rightarrow$ **Alveolar Parting** (lips parted, $\text{MAR} > 0.11$) $\rightarrow$ Emits **`N`**.
  * **`C` vs `O`**:
    * Hold circular hand with mouth round puckered ($\text{MAR} > 0.38$) $\rightarrow$ Emits **`O`**.
    * Hold open aperture hand with neutral/wide mouth $\rightarrow$ Emits **`C`**.
  * **`D` vs `1`**:
    * Point index upright while tapping tongue/lips $\rightarrow$ Emits **`D`**.
    * Point index upright with neutral facial posture $\rightarrow$ Emits **`1`**.

---

### Innovation 3: Ambient Peripheral Vision HUD
* **Purpose**: Allows deaf individuals to receive AI recognition feedback in their $140^\circ$ peripheral vision without ever breaking eye contact with their conversation partner.
* **How to Enable**: In the sidebar under **💡 AI Innovations**, toggle **`🌈 Ambient Peripheral Vision HUD`** *(Enabled by default)*.
* **Halo Color Meanings**:
  * <span style="color:#0284C7; font-weight:bold;">Cyan Pulse</span>: Word or character committed successfully.
  * <span style="color:#F59E0B; font-weight:bold;">Amber Gold Shimmer</span>: Tracking in progress / low confidence margin.
  * <span style="color:#EF4444; font-weight:bold;">Crimson Pulse</span>: Typo detected, character deleted, or backspace registered.
  * <span style="color:#10B981; font-weight:bold;">Emerald Glow</span>: Whole-phrase or sentence completed.

---

### Innovation 4: One-Shot Biomechanical Hand Calibration
* **Purpose**: Personalizes the recognition models to your unique finger lengths, bone proportions, and resting hand tremor in 10 seconds.
* **How to Run**:
  1. In the sidebar, expand **`🖐️ Biomechanical Hand Calibration`**.
  2. Click **`✨ Auto-Tune Hand Calibration`**.
  3. Hold your hand open and relaxed toward the camera for 10 seconds.
  4. The engine records bone segment ratios (index-to-palm, middle-to-palm) and tunes the 1€ filter cutoff frequency to your exact tremor profile.
  5. Saved automatically to `data/user_hand_profile.json`. Click **`🔄 Reset to Canonical Proportions`** anytime to restore defaults.

---

### Innovation 5: Air-Workspace In-Meeting Gesture Macros
* **Purpose**: Control virtual meetings (Zoom, Microsoft Teams, Google Meet) seamlessly using spatial air gestures while presenting.
* **How to Enable**: In the sidebar, toggle **`✋ Air-Workspace Meeting Gestures`**.
* **Supported Spatial Shortcuts**:

```mermaid
flowchart LR
    A["Flat Palm Hold (1.2s)"] -->|"Toggles"| M["🎙️ Mute / Unmute Microphone"]
    B["Open Palm Flick Up"] -->|"Toggles"| H["✋ Raise / Lower Hand"]
    C["Two-Finger Swipe Right"] -->|"Sends VK_RIGHT"| N["➡️ Next Presentation Slide"]
    D["Two-Finger Swipe Left"] -->|"Sends VK_LEFT"| P["⬅️ Previous Presentation Slide"]
    E["Two-Handed Double Palm Tap"] -->|"clip.exe Sync"| T["📋 Copy Sentence to Clipboard"]
```

1. **Mute/Unmute Mic**: Hold your flat open palm toward the camera for 1.2 seconds.
2. **Raise/Lower Hand**: Make an open palm gesture and flick upward.
3. **Slide Navigation**: Extend index and middle fingers (two-finger pointer) and swipe rapidly to the right (Next Slide) or left (Previous Slide).
4. **Copy Transcript to Clipboard**: Bring both open palms together in a double-palm tap. The entire translated sentence is copied to your operating system clipboard with a toast notification.

---

## 5. Tab 2: Reverse Speech-to-Sign 3D Avatar (Two-Way Communication)

Empowers hearing colleagues or teachers to communicate back to deaf individuals via real-time 3D sign synthesis:

```
[ Hearing User Speaks or Types English ]
                 │
                 ▼
[ ASL Grammar Synthesizer & Gloss Translator ]
  - Strips English idioms, infers ASL topic-comment order
  - Resolves vocabulary to 70+ canonical word coordinates
  - Falls back to fingerspelling for names/unrecognized words
                 │
                 ▼
[ Interactive 3D Skeletal Avatar Engine ]
  - Full 360° orbital rotation, pan & zoom
  - Hermite biomechanical joint angle interpolation
  - Timeline playback scrubber (0.5x to 2.0x speed)
```

### How to Use the Avatar:
1. Navigate to **Tab 2: 🗣️➡️🤟 Reverse Speech-to-Sign Avatar**.
2. **Input Options**:
   * **Voice Input**: Click the **`🎙️ Speak Now`** button inside the avatar card and speak naturally into your microphone (powered by Web Speech API).
   * **Text Input**: Type any sentence into the text box (e.g. *"Hello, how are you today?"*) and click **`🤟 Animate ASL Signs`**.
3. **Interactive 3D Controls**:
   * **Click & Drag**: Rotate the 3D hand in 360° space to view tricky fingerspelling from any angle.
   * **Scroll Wheel**: Zoom in on fingertip articulation.
   * **Play / Pause**: Halt motion on any keyframe.
   * **Scrubber**: Drag the slider to inspect intermediate joint transitions.
   * **Speed Slider**: Select from `0.5x` (slow practice) up to `2.0x` (rapid conversational rate).

---

## 6. Tab 3: Static Image Analysis & Dataset Explorer

Ideal for evaluating single gestures, educational verification, or benchmarking without an active webcam:

1. Navigate to **Tab 3: 🖼️ High-Precision Gesture Image Analysis**.
2. **Upload Mode**:
   * Drag and drop any `.png`, `.jpg`, or `.jpeg` file containing a hand sign.
   * The system extracts skeletal landmarks, draws the dual-hand bounding box, displays the predicted sign, confidence percentage, and top-3 ranked alternatives.
   * Click **`➕ Add to Sentence`** or **`🔊 Speak Sign`** to integrate the result directly into your active transcript.
3. **Dataset Explorer Mode**:
   * Select any class from `0-9`, `A-Z`, `del`, `space`.
   * Click **`🎲 Load Random Dataset Sample`** to inspect ground-truth images and verify the model's spatial response.

---

## 7. Tab 4: Model Performance & Comprehensive Benchmarks

Displays live telemetry and offline audit reports:
* **Confusion Matrices**: Per-class precision, recall, and F1 scores for all 39 characters.
* **Inference Latency**:
  * PyTorch LSTM: $14.6\text{ ms}$ (68.4 FPS on CPU).
  * ONNX Runtime Engine: $6.2\text{ ms}$ (161 FPS on CPU).
  * SIMD JPEG Serialization: $1.24\text{ ms}$ (807 FPS throughput).
* **Anti-Tremor Dampening Score**: Verifies variance reduction ($98.7\%$) against synthetic Parkinsonian and physiological hand tremor.

---

## 8. Tab 5: Interactive Sign Tutorial & Practice Guide

Educational masterclass designed for beginners and non-signers:
1. **Interactive ASL Flashcards**: Select any letter or number to inspect:
   * Anatomical finger positioning instructions.
   * Thumb placement rules.
   * Common mistakes to avoid.
2. **Anatomical Invariants for Confused Cluster (`V`, `3`, `W`, `6`, `U`):**

| Sign | Upright Fingers | Ring Finger | Thumb State | Pinky Interaction | Mathematical Rule |
|---|---|---|---|---|---|
| **`V`** | Index + Middle (Spread) | **Curled** | **Folded into palm** | Curled | $d_{im} \ge 0.17$, Ring Curled, $d_{tp\_mcp} < 0.65$ |
| **`3`** | Index + Middle + Thumb | **Curled** | **Extended laterally** | Curled | Thumb Extended ($d_{tp\_mcp} \ge 0.80, d_{tw} \ge 0.75$), Ring Curled |
| **`W`** | Index + Middle + Ring (Spread) | **Extended** | Folded over pinky | Curled (Open Loop) | Ring Extended, $d_{tp} \ge 0.25$ or Spread ($d_{im}, d_{mr} \ge 0.18$) |
| **`6`** | Index + Middle + Ring | **Extended** | Touching pinky tip | **Pinched to Thumb Tip** | Ring Extended, Pinch Loop $d_{tp} < 0.25$ |
| **`U`** | Index + Middle (Touching) | **Curled** | **Folded into palm** | Curled | $d_{im} < 0.17$ (parallel stack), Ring Curled, Thumb Folded |

3. **Disambiguation Cheat-Sheets**:
   * **`V` vs `3`**: In **`V`**, the thumb is folded across the palm. In ASL **`3`**, the thumb is sticking OUT to the side.
   * **`V` vs `W`**: In **`V`**, only 2 fingers are upright (ring is curled). In **`W`**, 3 fingers are upright (ring is extended).
   * **`W` vs `6`**: In **`W`**, fingers form a spread fan and thumb does not touch pinky tip. In **`6`**, thumb and pinky touch tip-to-tip.
   * **`U` vs `V`**: Index and middle fingers pressed touching parallel (`U`) vs spread apart in a "V" (`V`).
   * **`C` vs `O`**: Open semi-circle gap $\ge 40\%$ (`C`) vs fingertips touching thumb in a closed loop (`O`).
   * **`Z` vs `1`**: Dynamic zigzag air-drawing stroke (`Z`) vs static vertical point hold (`1`).

---

## 9. Personal Vocabulary & Custom Terminology Manager

Located in the **Sidebar** under **📚 Personal Vocabulary & Custom Terms**:
* **Add Custom Words**: Type specialized terms (doctor names, medications like `METFORMIN`, corporate acronyms) and click **`➕ Add Word`**.
* **Personalized NLP Boosting**: When fingerspelling, the sentence engine's Levenshtein autocorrect and bigram autocomplete will automatically prioritize your custom terms.
* **Remove Terms**: Select a term from the dropdown and click **`🗑️ Remove`**.

---

## 10. Accessibility & Theme Customization

Located in the **Sidebar** under **♿ Accessibility & Visual Settings**:
* **High-Contrast Dark Theme**: Toggles a deep `#020617` background with `#FACC15` (high-visibility gold) accents for signers with low vision or photophobia.
* **Caption Font Sizes**:
  * *Small (1.2rem)*
  * *Medium (1.5rem)* — Standard default.
  * *Large (2.0rem)* — Clear classroom display.
  * *Extra Large (2.5rem)* — Auditorium / projection size.
* **Disable Logging**: Prevents predictions from being written to `data/predicted_signs.txt` for strict privacy compliance.

---

## 11. Troubleshooting & FAQ

### Q: Why is my camera screen black?
1. Check the **physical privacy slider** on your laptop above the lens.
2. In the sidebar under **Camera & Performance**, switch **Webcam Index** from `0` to **`1`** (Index 0 is often an infrared Windows Hello sensor).
3. Ensure **Capture Backend** is set to **`DirectShow`** or **`Auto`**.

### Q: Why is the webcam lagging at 1 FPS?
* Switch **Capture Backend** from `Default` or `MSMF` to **`DirectShow`**. DirectShow hardware-decodes MJPG streams with zero buffering delay.

### Q: How do I sign a Space?
* Method 1: Hold an open flat hand horizontally or at a slight angle across your chest.
* Method 2: Pause your signing hand for 1.2 seconds; the engine will automatically autocorrect the word and add a space.
* Method 3: Click the **`␣ Space`** button under the video feed.

### Q: How do I run the automated system test suite?
Open your terminal and run:
```bash
python tests/test_full_system.py
```
This runs 8 automated subsystem tests in under 7 seconds, verifying model weights, ONNX runtimes, CTC continuous streaming, ASL grammar, and SIMD camera encoding.
