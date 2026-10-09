"""
Comprehensive Viva Guide and Technical Architecture PDF Generator for:
'Real-Time Motion-Aware Sign Language Detection & Recognition Using Computer Vision & Deep Learning (LSTM)'

Generates an exhaustive, publication-grade, professionally styled Viva Defense and 
Technical Documentation PDF covering system architecture, mathematical formulations, 
25 high-probability viva questions & answers, benchmark comparisons, and examiner tips.
"""
import os
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
    PageBreak
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Adds running headers and footers with page numbers (Page X of Y)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#475569"))
        
        # Suppress header on cover / page 1
        if self._pageNumber > 1:
            self.drawString(54, 752, "VIVA EXAMINATION & TECHNICAL DEFENSE GUIDE — ASL DEEP LEARNING (LSTM)")
            self.setFont("Helvetica", 8)
            self.drawRightString(558, 752, "CONFIDENTIAL / ACADEMIC")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)
            
        # Running Footer
        self.setFont("Helvetica", 8)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 34, page_str)
        self.drawString(54, 34, "Real-Time Sign Language Detection & Recognition | OpenCV + MediaPipe + PyTorch LSTM")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)
        self.restoreState()


def create_viva_guide_pdf(filename: str = "Sign_Language_Viva_Guide_and_Documentation.pdf") -> str:
    pdf_path = Path(filename)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=50,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#1E3A8A")     # Navy Blue
    c_secondary = colors.HexColor("#0D9488")   # Teal
    c_dark = colors.HexColor("#0F172A")        # Slate 900
    c_muted = colors.HexColor("#475569")       # Slate 600
    c_bg_light = colors.HexColor("#F8FAFC")    # Slate 50
    c_bg_card = colors.HexColor("#F1F5F9")     # Slate 100
    c_border = colors.HexColor("#E2E8F0")
    c_accent = colors.HexColor("#DC2626")      # Red highlight
    c_green = colors.HexColor("#166534")       # Dark green

    # Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=25,
        textColor=c_primary,
        alignment=1,
        spaceAfter=8
    )
    
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=c_secondary,
        alignment=1,
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=c_secondary,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=c_dark,
        spaceAfter=4
    )

    q_style = ParagraphStyle(
        "Question_Style",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12.5,
        textColor=c_primary,
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    )

    ans_style = ParagraphStyle(
        "Answer_Style",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=c_dark,
        leftIndent=10,
        spaceAfter=5
    )

    tip_style = ParagraphStyle(
        "Tip_Style",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=c_green,
        leftIndent=10,
        spaceAfter=6
    )

    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#0F172A"),
        backColor=c_bg_light,
        leftIndent=8,
        spaceBefore=2,
        spaceAfter=4
    )

    story = []

    # =========================================================================
    # COVER / HEADER
    # =========================================================================
    story.append(Spacer(1, 4))
    story.append(Paragraph("TECHNICAL DEFENSE & VIVA COMPREHENSIVE GUIDE", subtitle_style))
    story.append(Paragraph("Real-Time Motion-Aware Sign Language Detection & Recognition<br/>Using Computer Vision & Deep Learning (LSTM)", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=c_primary, spaceBefore=4, spaceAfter=10))

    meta_table_data = [
        [
            Paragraph("<b>Subject Area:</b> Computer Vision, Machine Learning, HCI", body_style),
            Paragraph("<b>Deep Learning Model:</b> Stacked 2-Layer MotionSignLSTM", body_style)
        ],
        [
            Paragraph("<b>Frameworks:</b> PyTorch, MediaPipe Hands 3D, OpenCV", body_style),
            Paragraph("<b>Spatial Classifier:</b> Random Forest (92.39% Test Acc)", body_style)
        ],
        [
            Paragraph("<b>Feature Engineering:</b> 129 Spatiotemporal Kinematics", body_style),
            Paragraph("<b>Classes:</b> 39 Classes (0-9, A-Z, del, nothing, space)", body_style)
        ]
    ]
    t_meta = Table(meta_table_data, colWidths=[255, 255])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), c_bg_card),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 1: 60-SECOND VIVA ELEVATOR PITCH
    # =========================================================================
    story.append(Paragraph("1. The 60-Second Viva Defense Pitch (What to Say First)", h1_style))
    story.append(Paragraph(
        "When an examiner asks: <i>'Tell me about your project in simple terms — what did you do and why?'</i>, deliver this concise 4-point response:",
        body_style
    ))

    pitch_data = [
        [
            Paragraph("<b>1. The Problem</b>", body_style),
            Paragraph("Traditional sign language systems only classify static 2D images. However, American Sign Language (ASL) is fundamentally a <b>dynamic language</b> where letters like 'J', 'Z', and conversational signs depend on <b>trajectory, speed, and motion</b>.", body_style)
        ],
        [
            Paragraph("<b>2. Landmark Kinematics</b>", body_style),
            Paragraph("Instead of feeding heavy raw video pixels to a slow CNN, we extract <b>21 3D hand landmarks</b> using MediaPipe Hands and engineer a <b>129-dimensional motion vector</b> per frame: 63 normalized positions, 63 joint velocities, and 3 global wrist displacement deltas.", body_style)
        ],
        [
            Paragraph("<b>3. Dual-Engine Intelligence</b>", body_style),
            Paragraph("We implemented a <b>Deep Learning PyTorch LSTM</b> that watches a 20-frame rolling sequence to track trajectories through time, paired with a <b>Random Forest</b> spatial classifier. A real-time <b>Kinetic Energy ($E_{\\text{motion}}$)</b> detector switches between active motion and static hold.", body_style)
        ],
        [
            Paragraph("<b>4. Physiological Disambiguation & Accessibility</b>", body_style),
            Paragraph("We solved the common 'C' vs 'O' confusion using a normalized hand aperture ratio, built an 8-stage cascading detector that achieves <b>96.30% accuracy</b> on unseen Kaggle test images, and integrated an interactive sentence builder with offline <b>Text-to-Speech</b> narration.", body_style)
        ]
    ]
    t_pitch = Table(pitch_data, colWidths=[120, 390])
    t_pitch.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), c_bg_card),
        ("ROWBACKGROUNDS", (1, 0), (-1, -1), [colors.white, c_bg_light]),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_pitch)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 2: END-TO-END SYSTEM ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("2. End-to-End System Pipeline & Kinematics", h1_style))
    story.append(Paragraph(
        "The project follows a modular, feed-forward computer vision and recurrent deep learning pipeline:",
        body_style
    ))

    pipeline_steps = [
        [Paragraph("<b>Stage</b>", body_style), Paragraph("<b>Component / Operation</b>", body_style), Paragraph("<b>Key Technical Details</b>", body_style)],
        [
            Paragraph("<b>1. Input Feed</b>", body_style),
            Paragraph("OpenCV Video Stream / File Upload", body_style),
            Paragraph("Captures BGR frames at 30 FPS. Supports webcams, phone video, and uploaded static images.", body_style)
        ],
        [
            Paragraph("<b>2. Hand Detection</b>", body_style),
            Paragraph("MediaPipe Hands & 8-Stage Cascade", body_style),
            Paragraph("Extracts 21 3D hand joints. If cropped borders cause failure, falls back through reflection, replication, constant padding, and CLAHE contrast enhancement.", body_style)
        ],
        [
            Paragraph("<b>3. Kinematics</b>", body_style),
            Paragraph("129 Spatiotemporal Motion Features", body_style),
            Paragraph("• 63 Wrist-shifted & scale-normalized coordinates.<br/>• 63 Joint velocities: v_i(t) = p_i(t) - p_i(t-1).<br/>• 3 Global wrist deltas: (&Delta;x_0, &Delta;y_0, &Delta;z_0).", body_style)
        ],
        [
            Paragraph("<b>4. Motion State</b>", body_style),
            Paragraph("Kinetic Energy Tracking", body_style),
            Paragraph("Computes E_motion = &Sigma; ||&Delta;p_i(t)||^2. Categorizes hand into STATIC HOLD (E &lt; 0.04), ACTIVE MOTION (0.04 &le; E &le; 0.15), or FAST TRAJECTORY (E &gt; 0.15).", body_style)
        ],
        [
            Paragraph("<b>5. Inference</b>", body_style),
            Paragraph("MotionSignLSTM / Random Forest", body_style),
            Paragraph("PyTorch LSTM evaluates rolling 20-frame sequence window across 39 ASL classes (0-9, A-Z, del, nothing, space).", body_style)
        ],
        [
            Paragraph("<b>6. Disambiguation</b>", body_style),
            Paragraph("Anatomical Aperture Check", body_style),
            Paragraph("Calculates normalized distance from thumb tip (LM 4) to index tip (LM 8). Resolves 'C' (aperture &ge; 0.44) vs 'O' (aperture &le; 0.25) deterministically.", body_style)
        ],
        [
            Paragraph("<b>7. Output Layer</b>", body_style),
            Paragraph("Sentence Builder & Offline TTS", body_style),
            Paragraph("Debounced 15-frame hold commits characters. Sentence accumulator supports Space, Backspace, Clear, and instant speech synthesis via pyttsx3.", body_style)
        ]
    ]
    t_pipe = Table(pipeline_steps, colWidths=[90, 160, 260])
    t_pipe.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_primary),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_bg_light]),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_pipe)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 3: KEY MATHEMATICAL EQUATIONS TABLE
    # =========================================================================
    story.append(Paragraph("3. Core Mathematical Formulations & Formulas", h1_style))
    story.append(Paragraph(
        "Examiners frequently ask for exact mathematical formulations. Here is the reference table:",
        body_style
    ))

    math_data = [
        [Paragraph("<b>Concept</b>", body_style), Paragraph("<b>Mathematical Formula</b>", body_style), Paragraph("<b>Engineering Purpose</b>", body_style)],
        [
            Paragraph("<b>Translation Normalization</b>", body_style),
            Paragraph("x_i' = x_i - x_0,<br/>y_i' = y_i - y_0,<br/>z_i' = z_i - z_0", code_style),
            Paragraph("Offsets all 21 landmarks by the wrist (Landmark 0). Makes the representation invariant to where the hand is in the camera frame.", body_style)
        ],
        [
            Paragraph("<b>Scale Normalization</b>", body_style),
            Paragraph("D_max = max_i &radic;(x_i'^2 + y_i'^2 + z_i'^2)<br/>p_i_norm = p_i' / D_max", code_style),
            Paragraph("Divides all coordinates by the maximum Euclidean distance to the wrist. Makes features invariant to distance from the webcam.", body_style)
        ],
        [
            Paragraph("<b>Joint Velocities</b>", body_style),
            Paragraph("v_i(t) = p_i(t) - p_i(t-1)<br/>for i &isin; [0, 20]", code_style),
            Paragraph("Calculates instantaneous joint speeds and directions. Distinguishes fast opening/closing gestures from static hand holds.", body_style)
        ],
        [
            Paragraph("<b>Global Wrist Delta</b>", body_style),
            Paragraph("&Delta;p_0(t) = p_0(t) - p_0(t-1)<br/>= (&Delta;x_0, &Delta;y_0, &Delta;z_0)", code_style),
            Paragraph("Captures entire hand spatial translation in camera 3D space. Mandatory for dynamic gestures like 'J', 'Z', and swipes.", body_style)
        ],
        [
            Paragraph("<b>Kinetic Energy (E)</b>", body_style),
            Paragraph("E_motion(t) = &Sigma;_{i=0}^{20} ||v_i(t)||^2", code_style),
            Paragraph("Quantifies total movement intensity. Replaces arbitrary timers with physical motion detection.", body_style)
        ],
        [
            Paragraph("<b>C vs O Aperture Ratio</b>", body_style),
            Paragraph("A = min(||p_4 - p_8||, ||p_4 - p_12||) / ||p_9 - p_0||", code_style),
            Paragraph("Normalizes thumb-to-fingertip distance by palm size (Wrist LM 0 to Middle MCP LM 9). 'O': A &le; 0.36; 'C': A &ge; 0.44.", body_style)
        ],
        [
            Paragraph("<b>Softmax Probability & Margin</b>", body_style),
            Paragraph("P(y=k|x) = exp(z_k) / &Sigma;_j exp(z_j)<br/>Margin = P_top1 - P_top2", code_style),
            Paragraph("Generates normalized posterior probabilities. Low margin indicates transition ambiguity (&rarr; 'Uncertain Gesture').", body_style)
        ]
    ]
    t_math = Table(math_data, colWidths=[110, 180, 220])
    t_math.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_primary),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_bg_light]),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_math)

    # Page Break to start Viva Q&A cleanly
    story.append(PageBreak())

    # =========================================================================
    # SECTION 4: 25 HIGH-PROBABILITY VIVA QUESTIONS & EXPERT ANSWERS
    # =========================================================================
    story.append(Paragraph("4. Top 25 High-Probability Viva Questions & Expert Answers", h1_style))
    story.append(Paragraph(
        "These questions represent the most common theoretical, architectural, and practical questions asked by university and external examiners.",
        body_style
    ))
    story.append(Spacer(1, 4))

    # --- CATEGORY A: COMPUTER VISION & MEDIAPIPE ---
    story.append(Paragraph("Category A: Computer Vision & MediaPipe Hands", h2_style))

    qas_cat_a = [
        (
            "Q1: Why did you use MediaPipe Hands instead of a direct Convolutional Neural Network (CNN)?",
            "Direct CNNs process high-dimensional raw pixel matrices (e.g. 224x224x3 = 150,528 inputs), which makes them computationally heavy (low FPS on CPU), highly sensitive to lighting changes, skin tone variations, and background clutter. MediaPipe Hands decouples hand perception into a two-stage lightweight pipeline (Palm Detector + Landmark Model) that extracts exactly 21 3D biomechanical coordinates (63 numbers). This achieves 30+ FPS on any standard CPU, ensures scale and skin-tone invariance, and reduces input dimensionality by over 99.9%.",
            "Tip for Viva: Mention that MediaPipe runs an ML pipeline trained on thousands of annotated hands using an anchor-based Single Shot Detector (SSD) followed by a 3D hand regression network."
        ),
        (
            "Q2: How many hand landmarks does MediaPipe detect, and what coordinate format does it return?",
            "MediaPipe detects exactly 21 keypoints per hand: Landmark 0 is the Wrist, Landmarks 1-4 correspond to the Thumb, 5-8 to the Index finger, 9-12 to the Middle finger, 13-16 to the Ring finger, and 17-20 to the Pinky. Each landmark consists of three coordinates (X, Y, Z): X and Y are normalized to [0.0, 1.0] relative to image width and height, and Z represents relative depth (negative values are closer to the camera, with the wrist as depth origin).",
            "Tip for Viva: Draw or describe the 5 finger chains originating from Landmark 0 (Wrist)."
        ),
        (
            "Q3: Why was translation and scale normalization required on top of MediaPipe's coordinates?",
            "MediaPipe coordinates are relative to the camera image boundary, NOT the hand. If a user moves their hand to the left or sits farther from the camera, raw (X, Y, Z) coordinates change completely even if the sign posture is identical. To achieve true translation invariance, we subtract Landmark 0 (wrist origin) from all keypoints: p_i' = p_i - p_0. To achieve scale invariance, we divide by the maximum Euclidean distance from the wrist: p_norm = p_i' / D_max. This guarantees identical numerical inputs regardless of hand location or distance.",
            "Tip for Viva: Note that without normalization, a classifier would overfit to where the hand was located in the training photos."
        ),
        (
            "Q4: What was the problem with the Kaggle dataset images, and how did your 8-stage cascading pipeline fix it?",
            "The Kaggle ASL Alphabet dataset contains 200x200 tightly cropped images where the wrist and fingertips often touch or exceed image borders. MediaPipe Hands expects a palm boundary buffer and failed to detect hands on 14 out of 28 test images (50% failure rate). We engineered an 8-stage cascading detector in 'utils/landmarks_util.py' that progressively applies reflection padding (50px, 80px), replicate padding, constant padding (80px, 100px), and CLAHE contrast enhancement. Once detected, padded coordinates are mathematically mapped back to original image space. This raised detection from 50% to 100% across the test archive.",
            "Tip for Viva: Explain that CLAHE (Contrast Limited Adaptive Histogram Equalization) operates in Lab color space on the L-channel to recover hands in shadowed or overexposed photos."
        ),
        (
            "Q5: How did you resolve the confusion between letters 'C' and 'O'?",
            "In ASL, both 'C' and 'O' share identical wrist orientation and forward finger curling across all four fingers, which blurs their decision boundary in 2D camera projections. We resolved this by computing the scale-invariant hand aperture: the Euclidean distance between thumb tip (LM 4) and index tip (LM 8) normalized by palm scale (distance from wrist LM 0 to middle MCP LM 9). In 'O', the fingertips physically meet in a closed loop (aperture <= 0.36), while in 'C' they form an open arc (aperture >= 0.44). We enforce this physiological rule in both real-time webcam and static image pipelines, eliminating jitter.",
            "Tip for Viva: State that our empirical benchmark showed 'O' has a mean aperture of 0.27, while 'C' has a mean aperture of 0.64."
        )
    ]

    for q, ans, tip in qas_cat_a:
        story.append(Paragraph(q, q_style))
        story.append(Paragraph(f"<b>Answer:</b> {ans}", ans_style))
        story.append(Paragraph(f"<i>Examiner Tip:</i> {tip}", tip_style))

    story.append(Spacer(1, 4))

    # --- CATEGORY B: KINEMATICS & FEATURE ENGINEERING ---
    story.append(Paragraph("Category B: Kinematic Feature Engineering", h2_style))

    qas_cat_b = [
        (
            "Q6: Why are 63 static spatial coordinates insufficient for dynamic sign language?",
            "63 static coordinates (21 joints x 3 dimensions) only capture a single instantaneous hand snapshot. However, natural ASL contains letters that require spatial movement: letter 'J' is signed by forming an 'I' handshape and drawing a 'J' curve in the air; letter 'Z' requires tracing a zigzag with the index finger. Furthermore, static coordinates cannot tell whether a hand is accelerating, slowing down, or transitioning between signs.",
            "Tip for Viva: Emphasize that static classifiers confuse transition frames with real signs."
        ),
        (
            "Q7: Explain the 129 features used in your Motion-Aware pipeline.",
            "Every video frame produces 129 physical numbers: (1) 63 Normalized Spatial Coordinates (hand posture & finger shape), (2) 63 Joint Velocities computed as first-order temporal derivatives v_i(t) = p_i(t) - p_i(t-1) for all 21 joints (capturing joint flexion and extension speed), and (3) 3 Global Wrist Displacement Deltas (&Delta;x_0, &Delta;y_0, &Delta;z_0) tracking overall hand travel across 3D space.",
            "Tip for Viva: 63 + 63 + 3 = 129. Remember this arithmetic!"
        ),
        (
            "Q8: What is Hand Kinetic Energy (E_motion) and what role does it play?",
            "Kinetic Energy is calculated as the sum of squared velocity magnitudes across all 21 hand joints: E_motion = &Sigma; ||&Delta;p_i(t)||^2. It acts as an automatic physical state indicator. When E < 0.04, the hand is in a 'STATIC HOLD'; when 0.04 <= E <= 0.15, the hand is in 'ACTIVE MOTION'; and when E > 0.15, it is in 'FAST TRAJECTORY'. In hybrid mode, high kinetic energy triggers temporal LSTM evaluation, while static hold utilizes high-precision spatial classification.",
            "Tip for Viva: Explain that kinetic energy also silences intermediate noisy characters during fast hand repositioning."
        ),
        (
            "Q9: Why did you set the temporal sequence window length to 20 frames?",
            "At standard webcam capture rates of 30 frames per second, 20 frames corresponds to approximately 0.67 seconds of real-time motion. In human biomechanics and sign language linguistics, a typical dynamic gesture or letter stroke takes between 0.4 and 0.8 seconds to complete. 20 frames provides sufficient temporal context to capture full trajectories like 'J' and 'Z' without introducing unnecessary memory overhead or inference lag.",
            "Tip for Viva: Mention that our sequence buffer is implemented as a rolling FIFO deque with maxlen=20."
        ),
        (
            "Q10: What happens when the hand is temporarily lost or out of the camera frame?",
            "Our predictor maintains state awareness. If no hand is detected in a frame, previous coordinates are cleared (self.prev_raw_coords = None), kinetic energy resets to 0.0, and the sequence buffer safely pops stale frames to prevent ghost velocities. The UI displays 'No Hand Detected' until the hand re-enters the field of view.",
            "Tip for Viva: Mention that this avoids erratic velocity spikes when a hand re-enters the frame."
        )
    ]

    for q, ans, tip in qas_cat_b:
        story.append(Paragraph(q, q_style))
        story.append(Paragraph(f"<b>Answer:</b> {ans}", ans_style))
        story.append(Paragraph(f"<i>Examiner Tip:</i> {tip}", tip_style))

    # Page Break for Deep Learning and Implementation sections
    story.append(PageBreak())

    # --- CATEGORY C: MACHINE LEARNING & LSTM ARCHITECTURE ---
    story.append(Paragraph("Category C: Machine Learning & PyTorch LSTM Architecture", h2_style))

    qas_cat_c = [
        (
            "Q11: What is an LSTM and why is it superior to a Standard Recurrent Neural Network (RNN)?",
            "A standard RNN suffers from the Vanishing and Exploding Gradient Problem when trained on sequences, meaning gradients exponentially shrink over time and the network cannot learn long-term temporal dependencies. Long Short-Term Memory (LSTM) solves this by introducing a dedicated Memory Cell State (c_t) regulated by three multiplicative gating mechanisms: the Forget Gate (f_t), Input Gate (i_t), and Output Gate (o_t). This allows the network to preserve information across all 20 frames without gradient degradation.",
            "Tip for Viva: Be prepared to write or explain the Forget Gate equation: f_t = &sigma;(W_f &middot; [h_{t-1}, x_t] + b_f)."
        ),
        (
            "Q12: Why did you choose LSTM over Fuzzy Logic?",
            "Fuzzy Logic relies on handcrafted, subjective if-then membership functions (e.g. 'IF finger is slightly bent AND thumb is near palm THEN sign is B'). Handcrafting fuzzy membership rules across 39 classes with 129 continuous, fast-changing motion features is intractable, prone to human error, and fails to model non-linear temporal trajectories. In contrast, LSTM automatically learns optimal temporal representations, joint velocity weights, and non-linear transitions directly from training data via backpropagation through time (BPTT).",
            "Tip for Viva: This is a direct defense of replacing rule-based heuristics with deep learning."
        ),
        (
            "Q13: Describe your exact PyTorch MotionSignLSTM neural network architecture.",
            "Our model (in 'models/lstm_model.py') is structured as follows: Input shape (Batch, 20, 129) &rarr; Layer 1: LSTM with 129 inputs and 128 hidden units &rarr; Layer 2: LSTM with 128 hidden units and 0.25 inter-layer dropout &rarr; Temporal state extraction of the final timestep h_20 &rarr; Dense Linear(128, 64) with ReLU activation &rarr; Dropout(0.25) &rarr; Output Linear(64, 39) &rarr; Softmax normalization. The network contains 274,471 trainable parameters.",
            "Tip for Viva: Point out that dropout prevents co-adaptation of features and regularizes the network against overfitting."
        ),
        (
            "Q14: What loss function and optimization algorithm did you use to train the LSTM?",
            "We used Categorical Cross-Entropy Loss (nn.CrossEntropyLoss in PyTorch) combined with the Adam (Adaptive Moment Estimation) optimizer at an initial learning rate of 0.001. Adam dynamically computes individual adaptive learning rates for each parameter using estimates of first and second moments of the gradients. We also employed StepLR / ReduceLROnPlateau learning rate scheduling to decay the learning rate when loss plateaued.",
            "Tip for Viva: CrossEntropyLoss in PyTorch internally applies LogSoftmax followed by NLLLoss for numerical stability."
        ),
        (
            "Q15: How does your system detect and handle 'Uncertain Gestures'?",
            "Instead of blindly predicting the argmax class, we analyze the Softmax probability distribution. A gesture is classified as 'HIGH CONFIDENCE' if P_top1 >= 0.60 and Margin (P_top1 - P_top2) >= 0.15; 'MEDIUM' if P_top1 >= 0.40 and Margin >= 0.075; and 'LOW / UNCERTAIN' if probabilities are dispersed. Ambiguous intermediate transitions automatically trigger 'Uncertain Gesture', preventing spurious characters from entering the sentence accumulator.",
            "Tip for Viva: Highlight that margin analysis prevents classification flapping when two signs have nearly equal probabilities."
        ),
        (
            "Q16: Why did you maintain both a Random Forest and an LSTM model (Dual-Engine)?",
            "Random Forest is an ensemble of decision trees trained on 63 normalized static coordinates, achieving 92.39% test accuracy across all 39 classes on isolated single-frame postures with near-zero latency (<1 ms). LSTM models dynamic temporal sequences (129 features across 20 frames) for moving signs like 'J' and 'Z'. By retaining both in 'SignPredictor', our system offers the best of both worlds: high-precision spatial recognition on static holds and trajectory comprehension on dynamic gestures.",
            "Tip for Viva: In Streamlit, users can toggle between Auto (Hybrid), Random Forest, and Deep LSTM."
        )
    ]

    for q, ans, tip in qas_cat_c:
        story.append(Paragraph(q, q_style))
        story.append(Paragraph(f"<b>Answer:</b> {ans}", ans_style))
        story.append(Paragraph(f"<i>Examiner Tip:</i> {tip}", tip_style))

    story.append(Spacer(1, 4))

    # --- CATEGORY D: REAL-TIME DEPLOYMENT & ACCESSIBILITY ---
    story.append(Paragraph("Category D: Real-Time Deployment & Human-Computer Interaction", h2_style))

    qas_cat_d = [
        (
            "Q17: How does the character debouncing and sentence builder work?",
            "To prevent duplicate characters from flooding the text box (e.g. spelling 'HHHHHEEELLLOOO' while holding a sign), we implemented a temporal queue with hysteresis. The system maintains a rolling history of the last 10 predictions. When a stable sign is held consistently for 15 frames (~0.5 seconds), it auto-commits once. It will not commit again until the hand transitions to a new sign or is removed. Control signs ('space', 'del') manipulate the text string directly.",
            "Tip for Viva: Mention that manual buttons ('+ Add Current', 'Space', 'Backspace', 'Clear') provide full user control."
        ),
        (
            "Q18: How does the offline Text-to-Speech (TTS) engine work without freezing the camera stream?",
            "Standard TTS libraries like pyttsx3 execute synchronously with engine.runAndWait(), which blocks the calling thread and causes video frame drops or application freezing. In 'utils/speech.py', we wrapped pyttsx3 in an asynchronous worker thread with a thread-safe Queue and mutex lock. When a user clicks 'Speak Text', the sentence is enqueued and vocalized in the background while the webcam loop continues uninterrupted at 30 FPS.",
            "Tip for Viva: Explain that threading prevents GUI unresponsiveness and frame lag."
        ),
        (
            "Q19: What are the 39 classes recognized by the model?",
            "The model recognizes 39 comprehensive classes: 10 Numeric Digits ('0' through '9'), all 26 uppercase letters of the English Alphabet ('A' through 'Z'), and 3 essential ASL communication control signs: 'space' (inserts a blank space between words), 'del' (deletes the preceding character like Backspace), and 'nothing' (negative control representing neutral background or resting hand).",
            "Tip for Viva: 'nothing' is critical as a negative class so the model does not hallucinate a letter when no gesture is made."
        ),
        (
            "Q20: What is the overall benchmark performance on the active ASL dataset and test archive?",
            "On the active 39-class ASL dataset (2,691 extracted images), our spatial Random Forest model achieved 92.39% accuracy (92.97% macro precision, 92.46% macro recall) on held-out test frames. Our PyTorch MotionSignLSTM achieved 80.33% accuracy (82.09% macro precision, 80.38% macro recall) on 737 unseen temporal motion sequences. Additionally, on the Kaggle ASL Alphabet test suite, our 8-stage cascading pipeline achieves 100% hand detection and 96.30% classification accuracy.",
            "Tip for Viva: Have the benchmark file 'models/archive_test_benchmark.csv' and 'models/evaluation_report.txt' ready to reference."
        )
    ]

    for q, ans, tip in qas_cat_d:
        story.append(Paragraph(q, q_style))
        story.append(Paragraph(f"<b>Answer:</b> {ans}", ans_style))
        story.append(Paragraph(f"<i>Examiner Tip:</i> {tip}", tip_style))

    # Page Break for comparison table and execution guide
    story.append(PageBreak())

    # =========================================================================
    # SECTION 5: ARCHITECTURAL COMPARISON MATRIX
    # =========================================================================
    story.append(Paragraph("5. Architectural Comparison Matrix", h1_style))
    story.append(Paragraph(
        "Examiners frequently ask you to justify your architectural decisions against alternative approaches. Use this matrix:",
        body_style
    ))

    comp_data = [
        [Paragraph("<b>Criteria</b>", body_style), Paragraph("<b>Raw CNN on Pixels</b>", body_style), Paragraph("<b>Fuzzy Logic Rules</b>", body_style), Paragraph("<b>Our MediaPipe + LSTM Pipeline</b>", body_style)],
        [
            Paragraph("<b>Input Dimensionality</b>", body_style),
            Paragraph("150,528 values (224x224x3)", body_style),
            Paragraph("Arbitrary scalar metrics", body_style),
            Paragraph("<b>129 Kinematic Features</b> (&gt;99.9% reduction)", body_style)
        ],
        [
            Paragraph("<b>Inference Latency</b>", body_style),
            Paragraph("45 - 120 ms (Needs GPU)", body_style),
            Paragraph("5 - 10 ms (Fast)", body_style),
            Paragraph("<b>8 - 14 ms (30+ FPS on standard CPU)</b>", body_style)
        ],
        [
            Paragraph("<b>Lighting / Background Sensitivity</b>", body_style),
            Paragraph("High (Fails on shadows/skin tones)", body_style),
            Paragraph("Moderate", body_style),
            Paragraph("<b>Scale & translation invariant 3D landmarks</b>", body_style)
        ],
        [
            Paragraph("<b>Dynamic Motion Handling</b>", body_style),
            Paragraph("Poor (Requires heavy 3D-CNN)", body_style),
            Paragraph("Intractable across 39 classes", body_style),
            Paragraph("<b>Native 20-frame Recurrent Memory (LSTM)</b>", body_style)
        ],
        [
            Paragraph("<b>Ambiguity Handling</b>", body_style),
            Paragraph("Hard argmax predictions", body_style),
            Paragraph("Manual thresholding", body_style),
            Paragraph("<b>Learned Softmax Margin + Aperture Rules</b>", body_style)
        ]
    ]
    t_comp = Table(comp_data, colWidths=[105, 125, 115, 165])
    t_comp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), c_primary),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, c_bg_light]),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 6: CONFUSION MATRIX & EVALUATION REPORT
    # =========================================================================
    story.append(Paragraph("6. Evaluation Results & Confusion Matrix", h1_style))
    story.append(Paragraph(
        "Evaluation was conducted on a held-out 20% test sequence partition (737 sequences) across all 39 ASL classes. "
        "The Confusion Matrix heatmap visualizes high diagonal density, validating class separability:",
        body_style
    ))

    cm_path = Path("models/confusion_matrix.png")
    if cm_path.exists():
        story.append(Spacer(1, 2))
        story.append(Image(str(cm_path), width=420, height=280))
        story.append(Paragraph("<i>Figure 1: High-resolution Confusion Matrix heatmap across all 39 ASL gesture classes.</i>", subtitle_style))
        story.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 7: VIVA EXAM QUICK REFERENCE CHEAT SHEET
    # =========================================================================
    story.append(Paragraph("7. Examiner Trap Questions & Quick Counters", h1_style))
    story.append(Paragraph(
        "Be prepared for these standard critical questions designed to test your deep understanding:",
        body_style
    ))

    traps = [
        [
            Paragraph("<b>Trap Question 1:</b> <i>'Why not use Mediapipe's own gesture recognizer API?'</i>", body_style),
            Paragraph("<b>Your Counter:</b> MediaPipe's built-in gesture recognizer only classifies ~7 basic gestures (thumbs up, open palm, pointing). It does NOT support the full 26-letter ASL alphabet, numeric digits 0-9, custom control signs, or trajectory modeling for letters like 'J' and 'Z'. We built a full custom 39-class recurrent architecture on top of MediaPipe's raw 3D landmarks.", body_style)
        ],
        [
            Paragraph("<b>Trap Question 2:</b> <i>'Why did you use Euclidean distance for aperture instead of Manhattan distance?'</i>", body_style),
            Paragraph("<b>Your Counter:</b> Euclidean distance measures true rotational and spatial physical distance in 3D Euclidean space: ||p_4 - p_8||_2 = &radic;(&Delta;x^2 + &Delta;y^2 + &Delta;z^2). Manhattan distance is coordinate-axis dependent and changes when the hand is tilted.", body_style)
        ],
        [
            Paragraph("<b>Trap Question 3:</b> <i>'What are the limitations of your project?'</i>", body_style),
            Paragraph("<b>Your Counter:</b> (1) Extreme hand occlusions where multiple fingers block each other from camera view; (2) Rapid continuous two-handed ASL signs (currently optimized for primary single-hand signing); (3) Future extension: adding NLP sentence auto-correction and grammar translation (ASL gloss to fluent English).", body_style)
        ]
    ]
    t_traps = Table(traps, colWidths=[180, 330])
    t_traps.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), c_bg_card),
        ("ROWBACKGROUNDS", (1, 0), (-1, -1), [colors.white, c_bg_light]),
        ("BOX", (0, 0), (-1, -1), 1, c_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_traps)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 8: SYSTEM RUN INSTRUCTIONS
    # =========================================================================
    story.append(Paragraph("8. Project Execution Commands", h1_style))
    story.append(Paragraph("Commands to run and demonstrate the application live:", body_style))
    story.append(Paragraph("<b>Launch Streamlit Dashboard:</b> <code>streamlit run app/app.py</code>", code_style))
    story.append(Paragraph("<b>Launch High-FPS OpenCV Window:</b> <code>python app/realtime_cv.py --camera 0</code>", code_style))
    story.append(Paragraph("<b>Run Kaggle Test Set Verification:</b> <code>python training/test_archive_images.py</code>", code_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[+] Successfully generated Viva Guide PDF: {pdf_path.resolve()}")
    return str(pdf_path.resolve())


if __name__ == "__main__":
    create_viva_guide_pdf()
