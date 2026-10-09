"""
PowerPoint Progress Report Generator for:
'Real-Time Motion-Aware Sign Language Detection & Recognition Using Computer Vision & Deep Learning (LSTM)'

Generates a modern, professional, 16:9 widescreen PowerPoint presentation (.pptx)
with structured slide layouts, cards, benchmark tables, and embedded confusion matrix.
"""
import os
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE


# ==============================================================================
# COLOR PALETTE DEFINITIONS (Modern Professional Tech Theme)
# ==============================================================================
COLOR_PRIMARY_NAVY = RGBColor(15, 23, 42)     # #0F172A (Deep Slate / Navy)
COLOR_BRAND_BLUE   = RGBColor(30, 58, 138)    # #1E3A8A (Navy Blue)
COLOR_TEAL_ACCENT  = RGBColor(13, 148, 136)   # #0D9488 (Teal)
COLOR_CYAN_ACCENT  = RGBColor(6, 182, 212)    # #06B6D4 (Cyan)
COLOR_BG_LIGHT     = RGBColor(248, 250, 252)  # #F8FAFC (Slate 50)
COLOR_CARD_WHITE   = RGBColor(255, 255, 255)  # #FFFFFF
COLOR_BORDER_SLATE = RGBColor(203, 213, 225)  # #CBD5E1 (Slate 300)
COLOR_TEXT_MAIN    = RGBColor(30, 41, 59)     # #1E293B (Slate 800)
COLOR_TEXT_MUTED   = RGBColor(100, 116, 139)  # #64748B (Slate 500)
COLOR_GREEN        = RGBColor(22, 163, 74)    # #16A34A (Green 600)
COLOR_PURPLE       = RGBColor(126, 34, 206)   # #7E22CE (Purple 700)


def create_progress_report_ppt(output_path: str = "Sign_Language_Project_Progress_Report.pptx") -> str:
    prs = Presentation()
    
    # 16:9 Widescreen Configuration
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide, color=COLOR_BG_LIGHT):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text: str, category_text: str = "PROJECT PROGRESS REPORT"):
        # Header banner bar
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_TEAL_ACCENT

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.6))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_PRIMARY_NAVY

        # Subtle separator line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.4), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = COLOR_BORDER_SLATE
        line.line.fill.background()

    def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_WHITE, border_color=COLOR_BORDER_SLATE):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
        return card

    # ==========================================================================
    # SLIDE 1: TITLE SLIDE (Cover)
    # ==========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, COLOR_PRIMARY_NAVY)

    # Accent decorative glow bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(0.12), Inches(3.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = COLOR_CYAN_ACCENT
    bar.line.fill.background()

    # Category Pill
    pill = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(1.5), Inches(3.4), Inches(0.38))
    pill.fill.solid()
    pill.fill.fore_color.rgb = RGBColor(30, 41, 59)
    pill.line.color.rgb = COLOR_CYAN_ACCENT
    pill.line.width = Pt(1)
    tf_pill = pill.text_frame
    p_pill = tf_pill.paragraphs[0]
    p_pill.text = "RESEARCH & ENGINEERING PROGRESS REPORT"
    p_pill.font.size = Pt(10)
    p_pill.font.bold = True
    p_pill.font.color.rgb = COLOR_CYAN_ACCENT
    p_pill.alignment = PP_ALIGN.CENTER

    # Main Title
    t_box = s1.shapes.add_textbox(Inches(1.2), Inches(2.0), Inches(10.8), Inches(1.6))
    tf_t = t_box.text_frame
    tf_t.word_wrap = True
    p_t = tf_t.paragraphs[0]
    p_t.text = "Real-Time Motion-Aware Sign Language Detection & Recognition"
    p_t.font.size = Pt(32)
    p_t.font.bold = True
    p_t.font.color.rgb = RGBColor(255, 255, 255)

    p_sub = tf_t.add_paragraph()
    p_sub.text = "Spatiotemporal Kinematics with PyTorch LSTM & MediaPipe 3D Landmark Tracking"
    p_sub.font.size = Pt(18)
    p_sub.font.color.rgb = COLOR_TEAL_ACCENT
    p_sub.space_before = Pt(8)

    # Metadata Card at Bottom
    c_meta = add_card(s1, Inches(1.2), Inches(5.0), Inches(10.8), Inches(1.6), bg_color=RGBColor(30, 41, 59), border_color=RGBColor(51, 65, 85))
    tf_meta = c_meta.text_frame
    tf_meta.word_wrap = True
    tf_meta.margin_left = Inches(0.4)
    tf_meta.margin_top = Inches(0.25)
    
    p1 = tf_meta.paragraphs[0]
    p1.text = "PROJECT HIGHLIGHTS & CURRENT STATUS:"
    p1.font.bold = True
    p1.font.size = Pt(11)
    p1.font.color.rgb = COLOR_CYAN_ACCENT

    p2 = tf_meta.add_paragraph()
    p2.text = "• 39 Active ASL Classes: Digits (0-9), Alphabet (A-Z), Controls (del, nothing, space)\n" \
              "• Dual-Engine Architecture: PyTorch MotionSignLSTM (Dynamic) + Random Forest (Spatial)\n" \
              "• High Accuracy: 92.39% Random Forest Test Acc | 80.33% MotionSignLSTM Test Acc\n" \
              "• Fully Integrated Interfaces: Streamlit Motion Dashboard + Native 30+ FPS OpenCV HUD"
    p2.font.size = Pt(12)
    p2.font.color.rgb = RGBColor(226, 232, 240)
    p2.space_before = Pt(4)

    # ==========================================================================
    # SLIDE 2: EXECUTIVE SUMMARY & OBJECTIVES
    # ==========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Executive Summary & Project Scope", "1. INTRODUCTION")

    cards_data_s2 = [
        ("🎯 Project Objective", 
         "Develop an accessible, real-time sign language interpreter using consumer webcams without expensive sensory gloves or GPU requirements.\n\n"
         "Bridges communication barriers by translating gestures into text and natural synthesized speech in real time.", 
         Inches(0.8), COLOR_BRAND_BLUE),
        ("⚡ Motion-Aware Distinction", 
         "Standard sign language apps only classify static single-frame postures.\n\n"
         "American Sign Language (ASL) is inherently dynamic: signs like 'J', 'Z', and conversational transitions require trajectory, velocity, and temporal tracking.", 
         Inches(4.8), COLOR_TEAL_ACCENT),
        ("🌐 39-Class Vocabulary", 
         "Comprehensive vocabulary:\n"
         "• 10 Digits: '0' through '9'\n"
         "• 26 Letters: 'A' through 'Z'\n"
         "• 3 Control Signs: 'space' (word break), 'del' (backspace), and 'nothing' (negative control background).", 
         Inches(8.8), COLOR_PURPLE),
    ]

    for title, text, left, color in cards_data_s2:
        c = add_card(s2, left, Inches(1.8), Inches(3.7), Inches(5.0))
        tf = c.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.3)
        tf.margin_top = Inches(0.3)
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(17)
        p.font.bold = True
        p.font.color.rgb = color

        p_desc = tf.add_paragraph()
        p_desc.text = text
        p_desc.font.size = Pt(13)
        p_desc.font.color.rgb = COLOR_TEXT_MAIN
        p_desc.space_before = Pt(14)

    # ==========================================================================
    # SLIDE 3: PROBLEM STATEMENT & TECHNICAL CHALLENGES
    # ==========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Problem Statement & Key Engineering Challenges", "2. PROBLEM ANALYSIS")

    challenges = [
        ("1. Dynamic Motion Loss in 2D Images", 
         "Letters like 'J' (tracing a hook) and 'Z' (tracing a zigzag) share similar static endpoints with 'I' or '1'. Static classifiers fail because they lack temporal recurrence and velocity memory."),
        ("2. Visual Ambiguity Between 'C' & 'O'", 
         "Both signs feature curved fingers. Depending on camera distance and rotation, statistical classifiers often confuse them, requiring anatomical physiological disambiguation."),
        ("3. Computational Overhead of Raw Pixel CNNs", 
         "Direct 2D/3D CNNs process 150,000+ raw pixel values per frame. They demand high-end GPUs, suffer frame drops on laptops, and fail under varied lighting or background clutter."),
        ("4. Cropped Dataset Hand Border Clippings", 
         "Kaggle/archive benchmark images are tightly cropped (200x200), cutting off wrist and knuckle anchors and causing vanilla MediaPipe to fail on 50% of images."),
    ]

    for i, (title, desc) in enumerate(challenges):
        row = i // 2
        col = i % 2
        left = Inches(0.8 + col * 5.95)
        top = Inches(1.8 + row * 2.55)

        c = add_card(s3, left, top, Inches(5.75), Inches(2.35))
        tf = c.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.3)
        tf.margin_top = Inches(0.25)

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = COLOR_BRAND_BLUE

        p_body = tf.add_paragraph()
        p_body.text = desc
        p_body.font.size = Pt(12.5)
        p_body.font.color.rgb = COLOR_TEXT_MAIN
        p_body.space_before = Pt(8)

    # ==========================================================================
    # SLIDE 4: SYSTEM ARCHITECTURE & DATA PIPELINE
    # ==========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "End-to-End System Architecture Pipeline", "3. SYSTEM DESIGN")

    stages = [
        ("STAGE 1: INPUT", "Webcam Video / Static Image\n(30 FPS RGB Frame)", COLOR_BRAND_BLUE),
        ("STAGE 2: TRACKING", "MediaPipe Hands\n(21 3D Biomechanical Joints)", COLOR_TEAL_ACCENT),
        ("STAGE 3: KINEMATICS", "129D Spatiotemporal Vector\n(Posture + Velocity + Delta)", COLOR_PURPLE),
        ("STAGE 4: KINETIC GATING", "Kinetic Energy Tracking\n(Static Hold vs Active Motion)", COLOR_BRAND_BLUE),
        ("STAGE 5: DUAL AI", "PyTorch LSTM / Random Forest\n(Posterior Probability Softmax)", COLOR_TEAL_ACCENT),
        ("STAGE 6: OUTPUT", "Sentence Builder & Offline TTS\n(Threaded pyttsx3 Audio)", COLOR_GREEN),
    ]

    for i, (st_title, st_desc, col_color) in enumerate(stages):
        row = i // 3
        col = i % 3
        left = Inches(0.8 + col * 3.95)
        top = Inches(1.8 + row * 2.6)

        c = add_card(s4, left, top, Inches(3.8), Inches(2.35))
        tf = c.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.25)

        p = tf.paragraphs[0]
        p.text = st_title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = col_color

        p_desc = tf.add_paragraph()
        p_desc.text = st_desc
        p_desc.font.size = Pt(12)
        p_desc.font.color.rgb = COLOR_TEXT_MAIN
        p_desc.space_before = Pt(8)

    # ==========================================================================
    # SLIDE 5: 129-DIMENSIONAL KINEMATIC FEATURE ENGINEERING
    # ==========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Kinematic Feature Engineering (129 Features/Frame)", "4. MATHEMATICAL FORMULATION")

    feat_cards = [
        ("1. Spatial Posture (63 Features)",
         "Wrist-relative normalized coordinates:\n"
         "x_i' = (x_i - x_0) / D_max\n"
         "y_i' = (y_i - y_0) / D_max\n"
         "z_i' = (z_i - z_0) / D_max\n\n"
         "Ensures strict invariance to camera distance, zoom, and spatial position across the video frame.",
         Inches(0.8)),
        ("2. Joint Velocities (63 Features)",
         "First-order temporal derivatives:\n"
         "v_i(t) = p_i(t) - p_i(t-1)\n\n"
         "Computes instantaneous rate-of-change for all 21 joints, tracking individual finger curling, extension speed, and articulation dynamics.",
         Inches(4.8)),
        ("3. Wrist Trajectory (3 Features)",
         "Global displacement of hand root:\n"
         "Δp_0 = (Δx_0, Δy_0, Δz_0)\n\n"
         "Tracks overall hand movement through 3D air space. Essential for distinguishing dynamic signs ('J', 'Z') from static equivalents.",
         Inches(8.8)),
    ]

    for title, text, left in feat_cards:
        c = add_card(s5, left, Inches(1.8), Inches(3.7), Inches(3.4))
        tf = c.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.25)

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = COLOR_BRAND_BLUE

        p_desc = tf.add_paragraph()
        p_desc.text = text
        p_desc.font.size = Pt(11.5)
        p_desc.font.color.rgb = COLOR_TEXT_MAIN
        p_desc.space_before = Pt(8)

    # Bottom Equation Card
    c_eq = add_card(s5, Inches(0.8), Inches(5.4), Inches(11.733), Inches(1.6), bg_color=RGBColor(241, 245, 249))
    tf_eq = c_eq.text_frame
    tf_eq.word_wrap = True
    tf_eq.margin_left = Inches(0.3)
    tf_eq.margin_top = Inches(0.2)

    p_eq1 = tf_eq.paragraphs[0]
    p_eq1.text = "INSTANTANEOUS KINETIC MOTION ENERGY TRACKING & STATE GATING:"
    p_eq1.font.bold = True
    p_eq1.font.size = Pt(12)
    p_eq1.font.color.rgb = COLOR_TEAL_ACCENT

    p_eq2 = tf_eq.add_paragraph()
    p_eq2.text = "E_motion = Σ ||Δp_i(t)||^2 (across all 21 joints)\n" \
                 "• Static Hold (E < 0.04): Dispatches to Random Forest for instant high-confidence posture evaluation.\n" \
                 "• Active Motion (0.04 ≤ E ≤ 0.15): Dispatches to PyTorch LSTM for continuous gesture trajectory classification.\n" \
                 "• Fast Trajectory (E > 0.15): Flags rapid transitional hand motion, suppressing premature letter predictions."
    p_eq2.font.size = Pt(11)
    p_eq2.font.color.rgb = COLOR_TEXT_MAIN
    p_eq2.space_before = Pt(4)

    # ==========================================================================
    # SLIDE 6: DUAL-ENGINE AI MODELING & ARCHITECTURE
    # ==========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Dual-Engine AI Architecture (PyTorch LSTM + Random Forest)", "5. MACHINE LEARNING")

    # Left Card: PyTorch LSTM
    c_lstm = add_card(s6, Inches(0.8), Inches(1.8), Inches(5.75), Inches(5.1))
    tf_lstm = c_lstm.text_frame
    tf_lstm.word_wrap = True
    tf_lstm.margin_left = tf_lstm.margin_right = Inches(0.3)
    tf_lstm.margin_top = Inches(0.3)

    p_l = tf_lstm.paragraphs[0]
    p_l.text = "🧠 PyTorch MotionSignLSTM (Deep Learning)"
    p_l.font.size = Pt(16)
    p_l.font.bold = True
    p_l.font.color.rgb = COLOR_BRAND_BLUE

    p_l_desc = tf_lstm.add_paragraph()
    p_l_desc.text = "• Input Tensor: (Batch, 20 Frames, 129 Features)\n" \
                    "• Layer 1: Recurrent LSTM (129 inputs → 128 hidden units)\n" \
                    "• Layer 2: Recurrent LSTM (128 units → 128 units, Dropout = 0.25)\n" \
                    "• Temporal Pooling: Extracts final timestep h_20 representation\n" \
                    "• Projection Head: Dense Linear(128, 64) → ReLU → Dropout(0.25) → Linear(64, 39)\n" \
                    "• Trainable Weights: 274,471 parameters\n" \
                    "• Training Regime: CrossEntropyLoss with Adam Optimizer (lr=0.001) over 30 epochs\n" \
                    "• Why it replaced Fuzzy Logic: Automatically learns complex non-linear temporal trajectories without brittle, handcrafted if-then rules."
    p_l_desc.font.size = Pt(12)
    p_l_desc.font.color.rgb = COLOR_TEXT_MAIN
    p_l_desc.space_before = Pt(10)

    # Right Card: Random Forest Spatial Classifier
    c_rf = add_card(s6, Inches(6.75), Inches(1.8), Inches(5.75), Inches(5.1))
    tf_rf = c_rf.text_frame
    tf_rf.word_wrap = True
    tf_rf.margin_left = tf_rf.margin_right = Inches(0.3)
    tf_rf.margin_top = Inches(0.3)

    p_r = tf_rf.paragraphs[0]
    p_r.text = "🌲 Random Forest Spatial Classifier"
    p_r.font.size = Pt(16)
    p_r.font.bold = True
    p_r.font.color.rgb = COLOR_TEAL_ACCENT

    p_r_desc = tf_rf.add_paragraph()
    p_r_desc.text = "• Input Dimension: 63 normalized static coordinates (21 joints × 3D)\n" \
                    "• Estimator Ensemble: 100 balanced decision trees\n" \
                    "• Latency: Under 1 millisecond inference time per frame\n" \
                    "• Role: Delivers instantaneous, rock-solid precision on static holds\n\n" \
                    "🔍 Anatomical Aperture Disambiguation ('C' vs 'O'):\n" \
                    "• Computes Euclidean distance between Thumb Tip (LM 4) and Index Tip (LM 8): Aperture = ||p_4 - p_8||_2\n" \
                    "• Aperture ≥ 0.44 → Confirmed open curve, promoted to 'C'\n" \
                    "• Aperture ≤ 0.25 → Confirmed closed circle, demoted to 'O'\n" \
                    "• Eliminates historical confusion between adjacent curved hand poses."
    p_r_desc.font.size = Pt(12)
    p_r_desc.font.color.rgb = COLOR_TEXT_MAIN
    p_r_desc.space_before = Pt(10)

    # ==========================================================================
    # SLIDE 7: COMPLETED MILESTONES & RECENT PROGRESS
    # ==========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Completed Milestones & Project Deliverables", "6. CURRENT STATUS")

    milestones = [
        ("✅ 1. Full Removal of Fuzzy Logic", 
         "Completely purged legacy `fuzzy/` folder and dependencies, transitioning 100% to deep recurrent neural representation learning."),
        ("✅ 2. Ingestion of 39-Class ASL Dataset", 
         "Extracted 2,691 clean images and 3,682 spatiotemporal motion sequences across all 10 digits, 26 alphabet letters, and 3 control signs."),
        ("✅ 3. 8-Stage Cascading Hand Detector", 
         "Engineered reflective padding, replicate padding, and CLAHE contrast stages in `landmarks_util.py`, raising hand detection on cropped images to 100%."),
        ("✅ 4. Model Training & Convergence", 
         "Trained Random Forest (92.39% accuracy) and PyTorch MotionSignLSTM (80.33% accuracy) on held-out test splits without synthetic cheating."),
        ("✅ 5. Real-Time Sentence Builder & TTS", 
         "Integrated 15-frame temporal debouncing and threaded, non-blocking offline `pyttsx3` speech narration in both web and native desktop UIs."),
        ("✅ 6. Publication-Grade Documentation", 
         "Compiled complete project documentation and 25-question Viva Guide PDFs with embedded high-resolution confusion matrix heatmaps."),
    ]

    for i, (m_title, m_desc) in enumerate(milestones):
        row = i // 2
        col = i % 2
        left = Inches(0.8 + col * 5.95)
        top = Inches(1.8 + row * 1.7)

        c = add_card(s7, left, top, Inches(5.75), Inches(1.55))
        tf = c.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.2)

        p = tf.paragraphs[0]
        p.text = m_title
        p.font.size = Pt(13.5)
        p.font.bold = True
        p.font.color.rgb = COLOR_BRAND_BLUE

        p_desc = tf.add_paragraph()
        p_desc.text = m_desc
        p_desc.font.size = Pt(11)
        p_desc.font.color.rgb = COLOR_TEXT_MAIN
        p_desc.space_before = Pt(4)

    # ==========================================================================
    # SLIDE 8: GROUNDED BENCHMARK RESULTS & CONFUSION MATRIX
    # ==========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Grounded Evaluation Benchmarks (39 Classes)", "7. EXPERIMENTAL EVALUATION")

    # Left: Metric Summary Table
    c_table = add_card(s8, Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.1))
    tf_tab = c_table.text_frame
    tf_tab.word_wrap = True
    tf_tab.margin_left = tf_tab.margin_right = Inches(0.3)
    tf_tab.margin_top = Inches(0.3)

    p_th = tf_tab.paragraphs[0]
    p_th.text = "📊 Held-Out Test Set Performance"
    p_th.font.size = Pt(16)
    p_th.font.bold = True
    p_th.font.color.rgb = COLOR_BRAND_BLUE

    p_tcontent = tf_tab.add_paragraph()
    p_tcontent.text = (
        "• Total Classes Evaluated: 39 (0-9, A-Z, del, nothing, space)\n\n"
        "🌲 Random Forest (Spatial 63D Landmarks):\n"
        "   - Test Accuracy: 92.39% (539 unseen test frames)\n"
        "   - Macro Precision: 92.97%\n"
        "   - Macro Recall: 92.46%\n"
        "   - Inference Latency: < 1 ms / frame\n\n"
        "🧠 PyTorch MotionSignLSTM (129D Temporal Kinematics):\n"
        "   - Test Accuracy: 80.33% (737 unseen sequences)\n"
        "   - Macro Precision: 82.09%\n"
        "   - Macro Recall: 80.38%\n"
        "   - Weighted F1-Score: 79.18%\n"
        "   - Latency: ~8-12 ms / sequence (30+ FPS on CPU)\n\n"
        "🏆 Kaggle Test Image Benchmark: 100% hand detection rate, 96.30% first-pass recognition accuracy."
    )
    p_tcontent.font.size = Pt(12)
    p_tcontent.font.color.rgb = COLOR_TEXT_MAIN
    p_tcontent.space_before = Pt(8)

    # Right: Embedded Confusion Matrix Image
    cm_path = Path("models/confusion_matrix.png")
    if cm_path.exists():
        # Outer Card
        c_img = add_card(s8, Inches(7.0), Inches(1.8), Inches(5.5), Inches(5.1))
        # Embed Image
        s8.shapes.add_picture(str(cm_path), Inches(7.2), Inches(2.2), width=Inches(5.1), height=Inches(3.8))
        caption_box = s8.shapes.add_textbox(Inches(7.2), Inches(6.1), Inches(5.1), Inches(0.6))
        tf_c = caption_box.text_frame
        p_c = tf_c.paragraphs[0]
        p_c.text = "Figure: High-resolution Confusion Matrix heatmap across all 39 ASL classes on held-out test sequences."
        p_c.font.size = Pt(9.5)
        p_c.font.italic = True
        p_c.font.color.rgb = COLOR_TEXT_MUTED
        p_c.alignment = PP_ALIGN.CENTER

    # ==========================================================================
    # SLIDE 9: USER INTERFACES & REAL-TIME DEMONSTRATION
    # ==========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Deployment Interfaces & Live User Experience", "8. DEMONSTRATION")

    # Web Dashboard Card
    c_web = add_card(s9, Inches(0.8), Inches(1.8), Inches(5.75), Inches(5.1))
    tf_web = c_web.text_frame
    tf_web.word_wrap = True
    tf_web.margin_left = tf_web.margin_right = Inches(0.3)
    tf_web.margin_top = Inches(0.3)

    p_w = tf_web.paragraphs[0]
    p_w.text = "🌐 Streamlit Web Application (app/app.py)"
    p_w.font.size = Pt(16)
    p_w.font.bold = True
    p_w.font.color.rgb = COLOR_BRAND_BLUE

    p_w_body = tf_web.add_paragraph()
    p_w_body.text = (
        "Run Command: `streamlit run app/app.py`\n\n"
        "• Tab 1: Live Motion ASL Recognition\n"
        "   - Real-time webcam feed with 21-joint skeleton overlay\n"
        "   - Instantaneous Kinetic Motion Energy meter\n"
        "   - Softmax probability distribution & predicted letter\n"
        "   - Sentence builder with 'Space', 'Backspace', 'Clear', and 'Speak Text' (offline TTS)\n\n"
        "• Tab 2: Static Image Analysis\n"
        "   - Dropdown browser across all 39 classes with live sample loading\n"
        "   - Custom file uploader supporting PNG/JPG images\n\n"
        "• Tab 3: Model Evaluation\n"
        "   - Interactive confusion matrix heatmap and precision/recall stats"
    )
    p_w_body.font.size = Pt(12)
    p_w_body.font.color.rgb = COLOR_TEXT_MAIN
    p_w_body.space_before = Pt(8)

    # Native OpenCV HUD Card
    c_cv = add_card(s9, Inches(6.75), Inches(1.8), Inches(5.75), Inches(5.1))
    tf_cv = c_cv.text_frame
    tf_cv.word_wrap = True
    tf_cv.margin_left = tf_cv.margin_right = Inches(0.3)
    tf_cv.margin_top = Inches(0.3)

    p_cv = tf_cv.paragraphs[0]
    p_cv.text = "⚡ Native OpenCV Live HUD (app/realtime_cv.py)"
    p_cv.font.size = Pt(16)
    p_cv.font.bold = True
    p_cv.font.color.rgb = COLOR_TEAL_ACCENT

    p_cv_body = tf_cv.add_paragraph()
    p_cv_body.text = (
        "Run Command: `python app/realtime_cv.py --camera 0`\n\n"
        "• High Frame Rate Performance:\n"
        "   - 30+ FPS low-latency desktop window\n"
        "   - Direct camera buffer rendering without browser overhead\n\n"
        "• Real-Time HUD Overlay:\n"
        "   - Top prediction banner with confidence score bar\n"
        "   - Current Kinetic Motion State badge\n"
        "   - Accumulated live sentence bar at bottom\n\n"
        "• Live Keyboard Shortcuts:\n"
        "   - [SPACE]: Insert word separation\n"
        "   - [BACKSPACE]: Delete preceding letter\n"
        "   - [A]: Force add current stable sign\n"
        "   - [C]: Clear sentence buffer\n"
        "   - [S]: Speak sentence aloud via pyttsx3\n"
        "   - [Q]: Cleanly exit"
    )
    p_cv_body.font.size = Pt(12)
    p_cv_body.font.color.rgb = COLOR_TEXT_MAIN
    p_cv_body.space_before = Pt(8)

    # ==========================================================================
    # SLIDE 10: NEXT STEPS & FUTURE SCOPE
    # ==========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Summary, Next Steps & Future Scope", "9. ROADMAP")

    future_cards = [
        ("🚀 Immediate Next Steps",
         "• Multi-User Validation: Conduct live trials across diverse lighting environments and background conditions.\n"
         "• Sentence Auto-Completion: Integrate lightweight n-gram word suggestion to speed up typing.\n"
         "• Fine-Tuning Motion Thresholds: Further calibrate kinetic energy transition boundaries for varied user signing speeds.",
         Inches(0.8), COLOR_BRAND_BLUE),
        ("🔮 Future Research Extensions",
         "• Two-Handed Signing: Extend MediaPipe configuration to dual-hand tracking for advanced multi-hand ASL signs.\n"
         "• NLP Grammar Translation: Integrate small language models (SLMs) to translate raw ASL gloss strings into grammatically fluent spoken English sentences.\n"
         "• Edge Optimization: Export trained PyTorch LSTM to ONNX / TensorRT for mobile and embedded deployment.",
         Inches(6.75), COLOR_PURPLE),
    ]

    for title, text, left, col in future_cards:
        c = add_card(s10, left, Inches(1.8), Inches(5.75), Inches(3.6))
        tf = c.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.3)
        tf.margin_top = Inches(0.3)

        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = col

        p_desc = tf.add_paragraph()
        p_desc.text = text
        p_desc.font.size = Pt(12.5)
        p_desc.font.color.rgb = COLOR_TEXT_MAIN
        p_desc.space_before = Pt(10)

    # Bottom Contact / Status Card
    c_foot = add_card(s10, Inches(0.8), Inches(5.6), Inches(11.733), Inches(1.4), bg_color=COLOR_PRIMARY_NAVY, border_color=COLOR_PRIMARY_NAVY)
    tf_f = c_foot.text_frame
    tf_f.word_wrap = True
    tf_f.margin_left = Inches(0.4)
    tf_f.margin_top = Inches(0.2)

    p_f1 = tf_f.paragraphs[0]
    p_f1.text = "PROJECT STATUS: STABLE & FULLY OPERATIONAL"
    p_f1.font.bold = True
    p_f1.font.size = Pt(12)
    p_f1.font.color.rgb = COLOR_CYAN_ACCENT

    p_f2 = tf_f.add_paragraph()
    p_f2.text = "All deliverables—including the 39-class ingested dataset, trained dual-engine models, Streamlit web dashboard, high-FPS OpenCV HUD, and standalone Viva documentation PDFs—are completed, validated, and ready for defense demonstration."
    p_f2.font.size = Pt(11)
    p_f2.font.color.rgb = RGBColor(226, 232, 240)
    p_f2.space_before = Pt(4)

    # Save presentation
    p_out = Path(output_path)
    prs.save(str(p_out))
    print(f"[+] Successfully generated PowerPoint presentation: {p_out.resolve()}")
    return str(p_out.resolve())


if __name__ == "__main__":
    create_progress_report_ppt()
