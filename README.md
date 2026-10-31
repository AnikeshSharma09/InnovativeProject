# Real-Time Continuous ASL Recognition & Linguistic Translation AI

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(Deep%20Learning)-red.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Accelerated%20CPU-orange.svg)](https://onnxruntime.ai/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic%20(Hands%2BFace%2BPose)-teal.svg)](https://developers.google.com/mediapipe)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange.svg)](https://streamlit.io/)

A multimodal, continuous American Sign Language (ASL) recognition and linguistic translation engine powered by **MediaPipe Holistic**, **CNN-BiLSTM-CTC sequence networks**, **ONNX Runtime**, and **rule-based ASL-to-English grammar translation**. Runs at 30+ FPS strictly on commodity CPU with 100% on-device privacy.

---

## 📌 Architectural Pillars & Engineering Highlights

### 1. Multimodal Perception & 210-D Feature Vector
- **Hands (129-D):** 63 wrist-relative landmarks, 63 instantaneous joint velocities, and 3 global wrist trajectory deltas.
- **Upper-Body Pose (32-D):** Shoulders, elbows, and wrists in 3D; bilateral elbow flexion angles; torso-relative hand positions; and signing-space elevation.
- **Non-Manual Facial Cues (18-D):** Eyebrow raise/furrow, eye openness, mouth aperture, 3D head yaw/pitch/roll, head nodding, and head shaking.
- **Bimanual Transforms (31-D):** 42-point inter-hand distance matrix, fingertip contact detection, and bilateral velocity symmetry ratio.
- **Scale-Invariant Normalization:** Normalized relative to shoulder midpoint and biacromial shoulder width, providing invariance to camera distance and user stature.

### 2. Continuous Stream Recognition (CTC)
- **Kinetic Energy Boundary Segmentation:** Monitors spatial velocity inflection points and kinetic hold apexes to segment fluid continuous signing without requiring artificial pauses.
- **CNN-BiLSTM-CTC Network:** A 1D temporal convolutional subnetwork followed by a 2-layer Bidirectional LSTM and CTC output projection head. Decodes variable-length sign sequences.
- **Temporal Non-Maximum Suppression (NMS):** Eliminates transitional co-articulation noise and duplicate sign emissions within rolling temporal windows.

### 3. ASL Grammar & Linguistic Translation
- **Topic-Comment to English SVO Inversion:** Resolves ASL spatial topic-comment phrasing into grammatically natural English word order.
- **Non-Manual Question Inversion:**
  - *WH-Questions:* Triggered by furrowed eyebrows (`eyebrow_furrow > 0.50`), inverting final WH-words to sentence-initial position (`"YOU GO WHERE?"` $\to$ `"Where are you going?"`).
  - *Yes/No Questions:* Triggered by raised eyebrows (`eyebrow_raise > 0.65`), prepending auxiliary verbs (`"YOU TIRED?"` $\to$ `"Are you tired?"`).
- **Head-Shake Negation:** Automatically converts affirmative verbs to negative form when accompanied by a head shake.
- **Temporal Marker Resolution:** Uses markers (`YESTERDAY`, `PAST`, `FINISH`) to conjugate verbs into proper past tense.

### 4. Advanced Bimanual Kinematics & Handedness Parity
- **42-Joint Bimanual Model:** Evaluates spatial distance between both hands, fingertip-to-palm contacts, and kinematic symmetry.
- **Handedness Normalization:** Automatically maps left-handed signers into canonical right-dominant space via horizontal axis mirroring ($X' = 2 \cdot X_{\text{mid}} - X$).

### 5. Confidence-Aware Inference & Personalization
- **Top-3 Candidate Ranking:** Returns top-3 ranked sign predictions with probability margins for every recognition event.
- **Uncertainty Gating:** Flags low-confidence signs as `UNCERTAIN` and displays one-click UI correction buttons.
- **Personal Calibration Prior:** Tracks user corrections in `data/user_corrections.json` and dynamically boosts posterior probabilities for difficult signs.
- **Personal Vocabulary Manager:** Users can add custom names, workplace terms, and medical vocabulary stored in `data/personal_vocabulary.json`.

### 6. High-Performance CPU Deployment
- **ONNX Runtime Acceleration:** Models exported to ONNX format with graph optimization and multi-threading, yielding **2.8x CPU speedup** over raw PyTorch.
- **100% On-Device Processing:** Zero cloud API dependencies. No video, audio, or telemetry is ever sent over the network.
- **Accessibility & Privacy Controls:** High-contrast color scheme, adjustable caption font size, and local file logging toggle.

---

## 📂 Project Directory Structure

```
InnovativeProject/
│
├── dataset/                        # ASL Alphabet training images (39 classes)
├── data/                           # Processed numerical sequence datasets & personal stores
│   ├── personal_vocabulary.json    # User custom dictionary terms & names
│   ├── user_corrections.json       # User override priors for calibration
│   └── predicted_signs.txt         # Persistent session transcript log
│
├── models/                         # Checkpoints & evaluation artifacts
│   ├── continuous_ctc.pth          # PyTorch CNN-BiLSTM-CTC sequence checkpoint
│   ├── continuous_ctc.onnx         # Accelerated ONNX Continuous CTC model
│   ├── sign_lstm.pth               # PyTorch MotionSignLSTM (129D)
│   ├── sign_lstm.onnx              # Accelerated ONNX LSTM model
│   ├── sign_model.pkl              # Spatial Random Forest baseline (99.3%)
│   ├── evaluation_report_v2.json   # Comprehensive benchmark metrics
│   └── benchmark_report.md         # Full Markdown benchmark report
│
├── utils/                          # Modular engineering libraries
│   ├── multimodal_features.py      # MediaPipe Holistic 210-D invariant feature extractor
│   ├── continuous_segmenter.py     # Kinetic inflection & temporal NMS segmenter
│   ├── asl_grammar.py              # ASL-to-English linguistic grammar translator
│   ├── bimanual_engine.py          # 42-landmark bimanual engine & handedness mirroring
│   ├── personalization.py          # Confidence manager, top-3 ranker, & vocabulary store
│   ├── onnx_exporter.py            # ONNX CPU session engine with thread pooling
│   └── sentence_engine.py          # Dual transcript engine with Levenshtein autocorrect
│
├── training/                       # Neural network training pipelines
│   ├── train_ctc.py                # Variable-length CTC network training & ONNX export
│   ├── benchmark_pipeline.py       # Full latency, CER/WER, & robustness benchmark harness
│   ├── train_lstm.py               # 129-D MotionSignLSTM training pipeline
│   └── evaluate_lstm.py            # Held-out 20% test split evaluation
│
├── prediction/                     # Real-time inference coordinators
│   ├── continuous_recognizer.py    # Continuous Holistic + CTC sequence coordinator
│   ├── predictor.py                # Motion-aware LSTM & Random Forest inference engine
│   └── camera_manager.py           # Robust multi-backend camera scanning with auto-fallback
│
├── app/                            # Web application & desktop UI
│   ├── app.py                      # Streamlit bilingual continuous & isolated app
│   └── realtime_cv.py              # Native OpenCV desktop HUD
│
├── TECHNICAL_REPORT.md             # In-depth architectural & mathematical technical report
├── requirements.txt                # Pinned dependencies (mediapipe==0.10.14, onnxruntime)
└── README.md                       # Comprehensive guide
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open Project
```powershell
cd d:\InnovativeProject
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

> **Note on Compatibility:** MediaPipe strictly requires `protobuf>=4.25.3,<5`. Our configuration pins `onnx==1.16.2` and `onnxruntime>=1.16.0` to ensure seamless runtime compatibility with MediaPipe 0.10.14 on Windows, macOS, and Linux.

---

## 🚀 Execution & Usage

### 1. Launch the Streamlit Web Application
```powershell
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser.

#### Key Features in the UI:
- **Continuous ASL Stream (CTC + Non-Manual):** Toggle in the left sidebar to activate continuous recognition with MediaPipe Holistic, non-manual facial cues, and ASL grammar translation.
- **Dual Bilingual Displays:**
  - *Raw ASL Glosses:* Displays gestures in spatial ASL topic-comment order.
  - *Translated English Sentence:* Displays natural English word order with question inversion, negation, and tense resolution.
- **Linguistic Badges:** Real-time badges for *WH-Questions*, *Yes/No Questions*, *Negation*, and *Statements*.
- **Top-3 Alternative Candidate Buttons:** One-click correction buttons that adapt the model's Bayesian calibration prior to your individual signing style.
- **Personal Vocabulary Manager:** Add custom names, medical terms, and domain phrases directly from the sidebar.
- **Accessibility & Privacy:** High-contrast mode, adjustable caption font size, and local file logging toggle.

### 2. Run Comprehensive Benchmarking Suite
To verify sequence CER/WER, latency percentiles (p50/p95/p99), and robustness metrics:
```powershell
python training/benchmark_pipeline.py
```
Results will be exported to `models/evaluation_report_v2.json` and `models/benchmark_report.md`.

### 3. Retrain the Continuous CTC Network
```powershell
python training/train_ctc.py --epochs 20 --batch-size 16 --export-onnx
```

### 4. Run Native High-FPS OpenCV Window
```powershell
python app/realtime_cv.py --camera 0
```

---

## 📊 Benchmark Summary

| Evaluation Dimension | Metric | Measured Value | Standard Target |
| :--- | :--- | :---: | :---: |
| **Sequence Accuracy** | Character Error Rate (CER) | **34.25%** | $< 40.0\%$ |
| | Word Error Rate (WER) | **36.67%** | $< 45.0\%$ |
| | Transition False-Positive Rate | **3.80%** | $< 5.0\%$ |
| | Sentence Accuracy | **63.3%** | $> 60.0\%$ |
| **Real-Time Latency** | Median Latency (p50) | **182.0 ms** | $< 250.0\text{ ms}$ |
| | 95th Percentile Latency (p95) | **283.6 ms** | $< 350.0\text{ ms}$ |
| | 99th Percentile Latency (p99) | **312.4 ms** | $< 400.0\text{ ms}$ |
| | Camera Capture Time | **5.0 ms** | $< 10.0\text{ ms}$ |
| | Landmark Perception Time | **13.5 ms** | $< 20.0\text{ ms}$ |
| | Feature Engineering Time | **1.2 ms** | $< 3.0\text{ ms}$ |
| **Robustness Scorecard** | Partial Occlusion (20% Dropout) | **91.4% retention** | $> 85.0\%$ |
| | Scale Invariance (0.6x to 1.5x) | **94.2% retention** | $> 90.0\%$ |
| | Rotation Tolerance (±25°) | **92.8% retention** | $> 85.0\%$ |
| | Left-Handed Mirroring Parity | **95.1% parity** | $> 90.0\%$ |

---

## 🔒 Privacy & On-Device Security

This software is built for high-security and private healthcare, educational, and enterprise settings:
- **Zero Cloud Network Transmission:** All video processing, landmark inference, CTC decoding, and grammar translation run strictly locally on the user's CPU.
- **Offline Text-to-Speech:** Pyttsx3 operates 100% offline without sending text to remote cloud synthesis APIs.
- **Audit-Ready Logging:** Prediction logging can be disabled at any time from the UI with zero persistent disk footprint.
