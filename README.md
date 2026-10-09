# Real-Time Motion-Aware Sign Language Detection and Recognition (Deep Learning LSTM)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%20(Deep%20Learning)-red.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands%203D-teal.svg)](https://developers.google.com/mediapipe)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange.svg)](https://streamlit.io/)

A complete, production-grade American Sign Language (ASL) detection and recognition system built using **OpenCV**, **MediaPipe Hands**, **PyTorch Deep Learning (LSTM)**, and **Streamlit**, supporting **39 comprehensive classes** (Digits `0`–`9`, Alphabet `A`–`Z`, and control gestures `del`, `nothing`, `space`).

---

## 📌 Project Overview

This project bridges human-computer interaction for the hearing and speech-impaired by translating dynamic hand movements, gestures, and static signs into digital text and spoken audio in real time.

### Key Engineering Features:
- **Biomechanical Hand Tracking**: Extracts 21 3D landmarks ($X, Y, Z$) using Google MediaPipe Hands.
- **39-Class Full ASL Coverage**: Comprehensive vocabulary covering 10 numeric digits (`0`–`9`), 26 alphabet letters (`A`–`Z`), and 3 functional control signs (`del`, `nothing`, `space`).
- **129-Dimensional Motion Feature Engineering**:
  1. **63 Spatial Posture Features**: Normalized coordinates relative to the wrist (scale & translation invariant).
  2. **63 Joint Velocities**: First-order temporal derivatives $v_i(t) = p_i(t) - p_i(t-1)$ capturing finger articulation speed and direction.
  3. **3 Global Wrist Trajectory Deltas**: Spatial path of the hand root in 3D camera space (vital for dynamic signs like 'J', 'Z', and spatial sweeps).
- **Instantaneous Kinetic Energy Tracking**: $E_{\text{motion}} = \sum_i \|\Delta p_i(t)\|^2$ automatically detects when active motion gestures begin and end.
- **Deep Recurrent Neural Network (LSTM)**: A 2-layer stacked PyTorch **MotionSignLSTM** network processes rolling sequence windows ($T = 20$ frames) to learn complex gesture trajectories.
- **Learned Temporal Confidence**: Replaces handcrafted fuzzy rules with the recurrent network's learned temporal hidden states, suppressing ambiguous or transitional motions as `"Uncertain Gesture"`.
- **Interactive Sentence Builder & Speech**: Debounced character accumulation with Space, Backspace, Clear, and non-blocking **pyttsx3 Text-to-Speech (TTS)** narration.
- **Dual User Interfaces**:
  1. **Streamlit Web Application** (`app/app.py`): Modern web dashboard featuring live webcam feed, kinetic motion meters, 39-class dataset browser & static image upload analysis, and confusion matrix analytics.
  2. **Native OpenCV Real-Time Window** (`app/realtime_cv.py`): High-FPS desktop HUD overlay with live keyboard hotkeys.

---

## 📂 Project Directory Structure

```
InnovativeProject/
│
├── dataset/                        # Raw ASL Alphabet images organized class-wise
│   ├── A/                          # e.g., dataset/A/001.jpg, dataset/A/002.jpg
│   ├── B/
│   └── ...
│
├── data/                           # Processed numerical sequence datasets
│   ├── X_sequences.npy             # 3D NumPy array (N x 20 frames x 129 features)
│   ├── y_sequences.npy             # Class labels
│   ├── X_test_seq.npy              # Held-out 20% test sequence features
│   ├── y_test_seq.npy              # Held-out 20% test sequence labels
│   └── landmarks.csv               # Backward-compatible tabular dataset
│
├── models/                         # Trained models, encoders, and evaluation results
│   ├── sign_lstm.pth               # PyTorch Deep Learning LSTM checkpoint
│   ├── sign_model.pkl              # Random Forest baseline checkpoint
│   ├── label_encoder.pkl           # Scikit-learn LabelEncoder instance
│   ├── lstm_metadata.json          # Architecture details and validation accuracy
│   ├── confusion_matrix.png        # High-res Seaborn confusion matrix heatmap (300 DPI)
│   ├── evaluation_results.json     # Grounded evaluation metrics
│   └── evaluation_report.txt       # Human-readable classification report
│
├── utils/
│   ├── __init__.py
│   ├── landmarks_util.py           # 129 motion feature extraction & normalization
│   ├── labels.py                   # ASL Alphabet classes, mappings, and display helpers
│   └── speech.py                   # Thread-safe pyttsx3 Text-to-Speech engine
│
├── training/
│   ├── __init__.py
│   ├── generate_sample_data.py     # Motion trajectory sequence generator
│   ├── train_lstm.py               # Phase 2: PyTorch LSTM training on sequences
│   ├── evaluate_lstm.py            # Phase 3: True test evaluation & confusion matrix
│   ├── extract_landmarks.py        # MediaPipe batch extraction from raw images
│   ├── clean_data.py               # Pandas data cleaning & validation
│   └── train_model.py              # Baseline Random Forest / SVM training
│
├── prediction/
│   ├── __init__.py
│   └── predictor.py                # Motion-aware LSTM inference engine
│
├── app/
│   ├── __init__.py
│   ├── app.py                      # Phase 4: Streamlit web application
│   └── realtime_cv.py              # Phase 4: Native OpenCV live webcam HUD (30+ FPS)
│
├── requirements.txt                # Pinned dependencies (torch, mediapipe==0.10.14)
├── Sign_Language_Project_Documentation.pdf # Full PDF documentation report
└── README.md                       # Comprehensive guide & viva defense
```

---

## ⚙️ Installation & Setup

### 1. Open Terminal in Project Directory
```powershell
cd d:\InnovativeProject
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 Execution Workflow

### Quickstart: Train & Test the Motion-Aware LSTM
```powershell
# 1. Generate 2,030 realistic motion sequence trajectories (20 frames x 129 features)
python training/generate_sample_data.py --samples-per-class 70 --sequence-length 20

# 2. Train the Deep Learning LSTM model (PyTorch, 25 epochs)
python training/train_lstm.py --epochs 25 --batch-size 32

# 3. Evaluate strictly on held-out 20% test sequences & generate Confusion Matrix
python training/evaluate_lstm.py

# 4. Launch the Streamlit Web Application
streamlit run app/app.py
```

---

### Phase 4 — Real-Time Applications

#### Option A: Interactive Streamlit Web Interface (Recommended for Demonstrations)
```powershell
streamlit run app/app.py
```
- Open `http://localhost:8501`.
- **Live Motion Webcam**: Tracks hand joint skeletons, displays predicted sign, LSTM probability, and **Kinetic Motion State** (`STATIC HOLD`, `ACTIVE MOTION`, `FAST MOTION`).
- **Sentence Builder**: Accumulate letters into words with Space, Backspace, Clear, and **Speak Text** (offline TTS).
- **Static Image Analysis Tab**: Upload photos to inspect landmarks and test prediction.
- **Model Evaluation Tab**: View live Confusion Matrix heatmap and precision/recall metrics.

#### Option B: Native OpenCV Real-Time Window (Ultra-Smooth 30+ FPS)
```powershell
python app/realtime_cv.py --camera 0
```
- Real-time video window with live HUD and keyboard controls:
  - `[SPACE]` : Space
  - `[BACKSPACE]` : Delete last letter
  - `[A]` : Manually add current stable letter
  - `[C]` : Clear sentence
  - `[S]` : Speak accumulated sentence via pyttsx3
  - `[Q]` : Quit

---

## 🧠 Kinematics & Deep Learning LSTM Architecture

### Motion Vectors (129 Features per Frame):
1. **Spatial Coordinates (63)**: Wrist-relative normalized positions $x_i', y_i', z_i'$.
2. **Joint Velocities (63)**: First-order temporal derivatives $v_i(t) = p_i(t) - p_i(t-1)$ across all 21 hand joints.
3. **Global Wrist Trajectory (3)**: 3D displacement vector $(\Delta x_0, \Delta y_0, \Delta z_0)$ tracking overall hand travel in the air.

### LSTM Network Specification:
- **Input Tensor**: `(Batch, 20, 129)`
- **Layer 1**: 129 inputs $\rightarrow$ 128 hidden recurrent units.
- **Layer 2**: 128 units $\rightarrow$ 128 hidden representations (Dropout = 0.25).
- **Projection Head**: Linear(128, 64) $\rightarrow$ ReLU $\rightarrow$ Dropout(0.25) $\rightarrow$ Linear(64, 39).
- **Parameters**: 274,471 trainable weights.

---

## 📊 Grounded Benchmark Results (39 Classes)

| Metric | Random Forest (Spatial 63D) | PyTorch MotionSignLSTM (129D Motion) |
| :--- | :---: | :---: |
| **Classes Evaluated** | 39 (`0`–`9`, `A`–`Z`, `del`, `nothing`, `space`) | 39 (`0`–`9`, `A`–`Z`, `del`, `nothing`, `space`) |
| **Input Shape** | `(63,)` static hand landmarks | `(20, 129)` spatiotemporal trajectory |
| **Test Accuracy** | **92.39%** (539 test samples) | **80.33%** (737 test sequences) |
| **Macro Precision** | **92.97%** | **82.09%** |
| **Macro Recall** | **92.46%** | **80.38%** |
| **Macro F1-Score** | **92.35%** | **79.26%** |
| **Inference Latency** | < 1 ms / frame | ~8–12 ms / sequence (30+ FPS on CPU) |

---

## 📥 Ingesting Custom Datasets & Scalability

The newly uploaded 39-class dataset (`asl_dataset` from archive) has already been synchronized to `dataset/`, extracted, and trained into both production models.

If you ever wish to train additional signs or custom recordings:
1. **Image Folders**: Place class folders inside `dataset/<new_class>/`.
2. **Pre-extracted Sequences**: Place sequence arrays at `data/X_sequences.npy` and `data/y_sequences.npy`.
3. Simply execute:
   ```powershell
   python training/extract_landmarks.py
   python training/train_model.py --model rf
   python training/train_lstm.py --epochs 30 --batch-size 32
   python training/evaluate_lstm.py
   python generate_viva_guide_pdf.py
   python generate_pdf_report.py
   ```
The pipeline will automatically adapt the output layer dimensions, retrain the models, regenerate the confusion matrix heatmap, and update all PDF technical reports!
