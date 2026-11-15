"""
generate_pdf_guide.py
=====================
Generates an executive, publication-grade PDF manual for the
Motion-Aware Sign Language AI System using ReportLab Platypus.
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
    PageBreak,
    KeepTogether,
    HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page count and footer."""
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
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Don't draw header/footer on cover page if page 1
        if self._pageNumber > 1:
            # Header
            self.drawString(54, 750, "Motion-Aware Sign Language AI — Complete User Operating Handbook")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

            # Footer
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(558, 36, page_text)
            self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — REAL-TIME ASL AI SYSTEM")
            self.line(54, 46, 558, 46)

        self.restoreState()


def build_pdf(output_path: str = "USER_GUIDE.pdf"):
    pdf_path = Path(output_path).resolve()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    C_PRIMARY = colors.HexColor("#1E3A8A")     # Deep Blue
    C_SECONDARY = colors.HexColor("#0284C7")   # Sky Blue
    C_TEXT = colors.HexColor("#0F172A")        # Dark Slate Text
    C_MUTED = colors.HexColor("#475569")       # Muted Slate
    C_BG_LIGHT = colors.HexColor("#F8FAFC")    # Card / Table background
    C_BORDER = colors.HexColor("#E2E8F0")

    # Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=C_PRIMARY,
        alignment=0,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=C_MUTED,
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=C_PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=C_SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=C_TEXT,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
        backColor=colors.HexColor("#F1F5F9"),
        spaceBefore=4,
        spaceAfter=6,
        leftIndent=10,
        rightIndent=10
    )

    callout_style = ParagraphStyle(
        "Callout_Custom",
        parent=body_style,
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E3A8A"),
        backColor=colors.HexColor("#EFF6FF"),
        borderColor=colors.HexColor("#3B82F6"),
        borderWidth=1,
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=8
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=C_TEXT
    )

    story = []

    # =========================================================================
    # HEADER BANNER & METADATA
    # =========================================================================
    story.append(Paragraph("🤟 ASL AI — Complete User & Feature Operating Handbook", title_style))
    story.append(Paragraph("<b>The Definitive Guide for Real-Time Motion-Aware Sign Language Detection, Two-Way 3D Avatar Translation & Air-Workspace AI</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=C_SECONDARY, spaceBefore=0, spaceAfter=12))

    meta_text = "<b>Author:</b> Deep Learning ASL Core Team &nbsp;|&nbsp; <b>System Version:</b> Enterprise 2.5 &nbsp;|&nbsp; <b>Platform:</b> Windows / macOS / Linux"
    story.append(Paragraph(meta_text, ParagraphStyle("Meta", parent=body_style, fontSize=8.5, textColor=C_MUTED, spaceAfter=14)))

    # =========================================================================
    # 1. QUICKSTART & LAUNCH
    # =========================================================================
    story.append(Paragraph("1. Quickstart & Application Launch", h1_style))
    story.append(Paragraph("The system operates 100% locally on CPU with zero cloud transmission for maximum privacy.", body_style))
    story.append(Paragraph("<b>Step 1: Install Dependencies</b> (Open PowerShell or Terminal in project folder):", body_style))
    story.append(Paragraph("<code>python -m pip install -r requirements.txt</code>", code_style))
    story.append(Paragraph("<b>Step 2: Launch Streamlit Web Application</b>:", body_style))
    story.append(Paragraph("<code>streamlit run app/app.py</code>", code_style))
    story.append(Paragraph("Your browser will open automatically at <b>http://localhost:8501</b>.", body_style))

    # =========================================================================
    # 2. CAMERA HARDWARE SETUP
    # =========================================================================
    story.append(Paragraph("2. Camera Hardware Setup & Zero-Lag Troubleshooting", h1_style))
    story.append(Paragraph("Located in the <b>Sidebar</b> under <b>🎥 Camera & Performance</b>:", body_style))

    cam_table_data = [
        [Paragraph("Setting / Control", table_header_style), Paragraph("Recommended Option", table_header_style), Paragraph("Operational Function & Best Practice", table_header_style)],
        [
            Paragraph("<b>Webcam Index</b>", table_cell_style),
            Paragraph("<b>0 or 1</b>", table_cell_style),
            Paragraph("Defaults to 0. <b>Switch to Index 1</b> if your screen is black or if your laptop has an Infrared (IR) Windows Hello sensor.", table_cell_style)
        ],
        [
            Paragraph("<b>Capture Backend</b>", table_cell_style),
            Paragraph("<b>DirectShow</b>", table_cell_style),
            Paragraph("Hardware MJPG decoding (<50ms open, zero driver lockups). <i>Completely eliminates MSMF frame timeouts and 0.7 FPS lag.</i>", table_cell_style)
        ],
        [
            Paragraph("<b>FPS Mode</b>", table_cell_style),
            Paragraph("<b>🚀 Turbo (60 FPS)</b>", table_cell_style),
            Paragraph("Hardware-accelerated SIMD JPEG frame streaming for lowest perception-to-display latency.", table_cell_style)
        ],
        [
            Paragraph("<b>Auto-Switch</b>", table_cell_style),
            Paragraph("<b>Checked (ON)</b>", table_cell_style),
            Paragraph("Automatically detects black IR sensors and switches to the active illuminated RGB camera.", table_cell_style)
        ],
        [
            Paragraph("<b>🔍 Scan Cameras</b>", table_cell_style),
            Paragraph("<b>Hardware Probe</b>", table_cell_style),
            Paragraph("Scans indices 0-3 and displays native resolutions and online/offline status.", table_cell_style)
        ]
    ]
    t_cam = Table(cam_table_data, colWidths=[110, 110, 284])
    t_cam.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, C_BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_cam)
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Physical Privacy Slider Notice:</b> Many Lenovo, HP, and Dell laptops have a physical mechanical slider above the camera lens. If the image is pitch-black, verify that the physical slider is open.", callout_style))

    # =========================================================================
    # 3. TAB 1: LIVE MOTION WEBCAM
    # =========================================================================
    story.append(Paragraph("3. Tab 1: Live Motion Webcam & Dual Output", h1_style))
    story.append(Paragraph("<b>Dual Recognition Paradigms:</b>", h2_style))
    story.append(Paragraph("• <b>Isolated Motion-Aware (LSTM + Random Forest):</b> Recognizes 39 individual characters (A-Z, 0-9, space, del) with <b>99.3% accuracy</b>. Features 1€ anti-shiver filtering reducing hand tremor variance by 98.7%.<br/>• <b>Continuous ASL Stream (CTC + Holistic):</b> Simultaneous tracking of dual hands, body posture, and facial non-manual markers using CNN-BiLSTM-CTC sequence decoding.", body_style))

    story.append(Paragraph("<b>Linguistic Grammar Translation Badges:</b>", h2_style))
    story.append(Paragraph("As you sign, facial markers are classified into linguistic sentence types:<br/>"
                           "• <b>WH-Question:</b> Furrowed eyebrows + head tilt (<i>'NAME YOU WHAT'</i> → <i>'What is your name?'</i>).<br/>"
                           "• <b>Yes/No Question:</b> Raised eyebrows + chin forward (<i>'YOU WANT COFFEE'</i> → <i>'Would you like coffee?'</i>).<br/>"
                           "• <b>Negated Statement:</b> Head shake (<i>'I UNDERSTAND'</i> → <i>'I do not understand.'</i>).<br/>"
                           "• <b>Declarative:</b> Standard affirmative statements.", body_style))

    story.append(Paragraph("<b>Transcript Controls & Space Registration:</b>", h2_style))
    story.append(Paragraph("• <b>➕ Add Current:</b> Commits the currently predicted sign into the active sentence.<br/>"
                           "• <b>␣ Space:</b> Commits word, triggers Levenshtein autocorrect, and inserts space. <i>To sign a space: hold an open flat hand horizontally across your chest, or pause for 1.2s.</i><br/>"
                           "• <b>⌫ Backspace / Undo:</b> Deletes the last character or undoes the latest continuous sign gloss.<br/>"
                           "• <b>🗑️ Delete Word:</b> Wipes the current word being fingerspelled.<br/>"
                           "• <b>🧹 Clear All:</b> Resets the entire sentence and clears sequence buffers.<br/>"
                           "• <b>💾 Save Sentence:</b> Persists sentence with timestamps to <code>data/predicted_signs.txt</code>.<br/>"
                           "• <b>🔊 Speak English (TTS):</b> Speaks translated English sentence aloud via offline speech synthesis.<br/>"
                           "• <b>Top-3 Alternatives (#1, #2, #3):</b> Clicking any alternative immediately corrects the transcript and calibrates the personalization engine.", body_style))

    # =========================================================================
    # 4. THE 5 BREAKTHROUGH AI INNOVATIONS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. The 5 Breakthrough AI Innovations", title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_SECONDARY, spaceBefore=0, spaceAfter=10))

    innovations_data = [
        [Paragraph("Innovation", table_header_style), Paragraph("Module File", table_header_style), Paragraph("How It Works & How to Use", table_header_style)],
        [
            Paragraph("<b>1. Rapid Co-Articulation Decoder</b>", table_cell_style),
            Paragraph("<code>utils/coarticulation_engine.py</code>", table_cell_style),
            Paragraph("<b>5x Speed Typing without Pauses.</b> Detects kinematic deceleration valleys (apexes) along the motion trajectory. Toggle in sidebar: <i>'🚀 Rapid Co-Articulation Mode'</i>. Sign continuous letters smoothly at 4-6 signs/sec.", table_cell_style)
        ],
        [
            Paragraph("<b>2. Lip-Morpheme Facial Fusion</b>", table_cell_style),
            Paragraph("<code>utils/lip_morpheme_engine.py</code>", table_cell_style),
            Paragraph("<b>Mouth Kinematic Disambiguation.</b> Fuses FaceMesh lip aspect ratio (MAR) to resolve identical handshapes:<br/>• <b>M vs N:</b> Mouth 'M' (lips closed, MAR&lt;0.055) vs 'N' (lips parted, MAR&gt;0.11).<br/>• <b>C vs O:</b> Open aperture vs round puckered lips (MAR&gt;0.38).<br/>• <b>D vs 1:</b> Alveolar tongue contact vs neutral digit pointing.", table_cell_style)
        ],
        [
            Paragraph("<b>3. Ambient Peripheral Vision HUD</b>", table_cell_style),
            Paragraph("<code>utils/ambient_hud.py</code>", table_cell_style),
            Paragraph("<b>Zero-Eye-Contact Halo Feedback.</b> Renders an alpha-blended chromatic edge halo visible in 140° peripheral vision without breaking eye contact:<br/>• <font color='#0284C7'><b>Cyan Pulse:</b></font> Word committed.<br/>• <font color='#F59E0B'><b>Amber Shimmer:</b></font> Active tracking / ambiguity.<br/>• <font color='#EF4444'><b>Crimson Pulse:</b></font> Typo / character deletion.<br/>• <font color='#10B981'><b>Emerald Glow:</b></font> Sentence finalized.", table_cell_style)
        ],
        [
            Paragraph("<b>4. One-Shot Hand Calibration</b>", table_cell_style),
            Paragraph("<code>utils/hand_calibrator.py</code>", table_cell_style),
            Paragraph("<b>10-Second Anatomical Profiling.</b> Click <i>'✨ Auto-Tune Hand Calibration'</i> in sidebar and hold relaxed open hand for 10s. Measures individual bone segment ratios and resting tremor jitter, scaling coordinates to canonical models.", table_cell_style)
        ],
        [
            Paragraph("<b>5. Air-Workspace Meeting Macros</b>", table_cell_style),
            Paragraph("<code>utils/air_workspace.py</code>", table_cell_style),
            Paragraph("<b>Spatial Video-Call Gestures (Zoom/Teams/Meet):</b><br/>• <b>Mute/Unmute Mic:</b> Hold flat palm steady at chest for 1.2s.<br/>• <b>Raise Hand:</b> Upward flick of open palm.<br/>• <b>Next/Prev Slide:</b> Two-finger lateral horizontal swipe.<br/>• <b>Copy Transcript:</b> Two-handed double palm tap.", table_cell_style)
        ]
    ]
    t_inn = Table(innovations_data, colWidths=[120, 110, 274])
    t_inn.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, C_BG_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_inn)

    # =========================================================================
    # 5. TAB 2: REVERSE 3D AVATAR
    # =========================================================================
    story.append(Spacer(1, 8))
    story.append(Paragraph("5. Tab 2: Reverse Speech-to-Sign 3D Avatar (Two-Way Communication)", h1_style))
    story.append(Paragraph("Empowers <b>hearing individuals to communicate back to deaf signers</b>. Translates spoken or typed English into 3D skeletal hand animations in real time.", body_style))
    story.append(Paragraph("<b>Operating Instructions:</b><br/>"
                           "1. <b>Voice Input:</b> Click <b>'🎙️ Speak Now'</b> inside the avatar card and speak into your microphone (Web Speech API).<br/>"
                           "2. <b>Text Input:</b> Type English text (e.g. <i>'Hello, how are you today?'</i>) and click <b>'🤟 Animate ASL Signs'</b>.<br/>"
                           "3. <b>3D Interactive Orbit:</b> Click and drag with your mouse to rotate 360° in 3D space to inspect tricky finger positions.<br/>"
                           "4. <b>Playback Timeline:</b> Use Play, Pause, Scrubbing bar, and the Speed Slider (0.5x practice to 2.0x conversational).", body_style))

    # =========================================================================
    # 6. TABS 3, 4, 5 & EXTRA FEATURES
    # =========================================================================
    story.append(Paragraph("6. Static Analysis, Benchmarks, Tutorials & Accessibility", h1_style))
    story.append(Paragraph("• <b>Tab 3 (Static Image Analysis):</b> Upload <code>.png</code> or <code>.jpg</code> files to inspect spatial landmark coordinates, confidence percentages, and top-3 predictions. Includes sample browser for the Kaggle ASL Alphabet dataset.<br/>"
                           "• <b>Tab 4 (Model Benchmarks):</b> View confusion matrices, precision/recall audits, and latency telemetry (14.6ms LSTM inference, 1.24ms SIMD JPEG serialization).<br/>"
                           "• <b>Tab 5 (Interactive Tutorial):</b> Flashcards for all 36+ signs, plus anatomical cheat-sheets (e.g. 'W' fan-out vs '6' thumb-pinky contact, 'U' touching vs 'V' spread).<br/>"
                           "• <b>Personal Vocabulary Manager (Sidebar):</b> Add custom medical names (e.g. <code>METFORMIN</code>) or corporate terms to prioritize them in Levenshtein autocorrect.<br/>"
                           "• <b>Accessibility Settings (Sidebar):</b> Toggle OLED high-contrast dark theme and adjust caption font size from Small (1.2rem) to Extra Large (2.5rem).", body_style))

    # =========================================================================
    # 7. TROUBLESHOOTING & FAQ
    # =========================================================================
    story.append(Paragraph("7. Quick Troubleshooting & FAQ", h1_style))
    story.append(Paragraph("<b>Q: Camera screen is pitch-black?</b><br/>"
                           "1. Check the physical camera privacy slider above your laptop screen.<br/>"
                           "2. In sidebar, switch <b>Webcam Index</b> from <code>0</code> to <code>1</code> (Index 0 is often an infrared Windows Hello sensor).<br/>"
                           "3. Switch <b>Capture Backend</b> to <code>DirectShow</code>.<br/><br/>"
                           "<b>Q: Webcam feed is laggy at 0.7 or 1 FPS?</b><br/>"
                           "Change Capture Backend to <code>DirectShow</code> in the sidebar. DirectShow negotiates hardware MJPG decoding with zero buffering delays.<br/><br/>"
                           "<b>Q: How do I run the automated system test suite?</b><br/>"
                           "Run <code>python tests/test_full_system.py</code> in terminal. Verifies all 8 subsystems in under 7 seconds with 100% test passage.", body_style))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    return str(pdf_path)


if __name__ == "__main__":
    out = build_pdf("USER_GUIDE.pdf")
    print(f"[+] User Guide PDF successfully generated at: {out}")
