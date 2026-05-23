import sys
import subprocess
import os
from pathlib import Path

# ── Ensure reportlab is installed ───────────────────────────────────────
try:
    import reportlab
except ImportError:
    print("reportlab not found. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "reportlab"])
    import reportlab

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Register Thai Fonts ────────────────────────────────────────────────
font_dir = Path("assets/fonts")
regular_font_path = font_dir / "Sarabun-Regular.ttf"
bold_font_path = font_dir / "Sarabun-Bold.ttf"
italic_font_path = font_dir / "Sarabun-Italic.ttf"

if not regular_font_path.exists():
    # Fallback to system fonts if google fonts are not found
    print("Warning: Google Sarabun font not found in assets/fonts. Trying system fallback...")
    # Mac system default
    system_font = Path("/System/Library/Fonts/Supplemental/Ayuthaya.ttf")
    if system_font.exists():
        pdfmetrics.registerFont(TTFont("Sarabun", str(system_font)))
        pdfmetrics.registerFont(TTFont("Sarabun-Bold", str(system_font)))
    else:
        # standard fallback
        pdfmetrics.registerFont(TTFont("Sarabun", "Helvetica"))
        pdfmetrics.registerFont(TTFont("Sarabun-Bold", "Helvetica-Bold"))
else:
    pdfmetrics.registerFont(TTFont("Sarabun", str(regular_font_path)))
    pdfmetrics.registerFont(TTFont("Sarabun-Bold", str(bold_font_path)))
    pdfmetrics.registerFont(TTFont("Sarabun-Italic", str(italic_font_path)))


def make_report(output_filename="outputs/OMNI_Project_Report.pdf"):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    # ── Define Styles ─────────────────────────────────────────────────────
    styles = getSampleStyleSheet()
    
    # Primary Colors
    c_primary = colors.HexColor("#1A365D")    # Deep Navy
    c_secondary = colors.HexColor("#2B6CB0")  # Slate Blue
    c_dark = colors.HexColor("#2D3748")       # Charcoal
    c_light = colors.HexColor("#EDF2F7")      # Warm White/Grey
    c_border = colors.HexColor("#CBD5E0")     # Light Grey
    c_success = colors.HexColor("#48BB78")    # Green
    
    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Sarabun-Bold',
        fontSize=24,
        leading=30,
        textColor=c_primary,
        alignment=1, # Center
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Sarabun',
        fontSize=14,
        leading=18,
        textColor=c_secondary,
        alignment=1, # Center
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Sarabun-Bold',
        fontSize=18,
        leading=22,
        textColor=c_primary,
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Sarabun-Bold',
        fontSize=13,
        leading=17,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Sarabun',
        fontSize=10,
        leading=14,
        textColor=c_dark,
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Sarabun',
        fontSize=10,
        leading=14,
        textColor=c_dark,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=5
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Sarabun',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2C3E50"),
        backColor=c_light,
        borderColor=c_border,
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=10
    )

    doc = SimpleDocTemplate(
        output_filename,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=50,
        bottomMargin=50
    )
    
    story = []
    
    # ── COVER PAGE ────────────────────────────────────────────────────────
    story.append(Spacer(1, 100))
    # Title
    story.append(Paragraph("รายงานโครงการพัฒนาโมเดลปัญญาประดิษฐ์ไทย", title_style))
    story.append(Paragraph("Thai Omni-Modal AI (OMNI)", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("การผสานระบบเสียง ภาพ และข้อความ สู่ปัญญาประดิษฐ์ระดับแนวหน้าสำหรับภาษาไทย", subtitle_style))
    
    story.append(Spacer(1, 120))
    
    # Metadata Block
    meta_data = [
        [Paragraph("<b>จัดทำโดย:</b> ทีมงานวิจัยและพัฒนา OMNI", body_style)],
        [Paragraph("<b>ผู้ร่วมพัฒนา AI:</b> Antigravity (Google DeepMind Team)", body_style)],
        [Paragraph("<b>วันที่มีผล:</b> 23 พฤษภาคม 2026", body_style)],
        [Paragraph("<b>สถานะโครงการ:</b> เฟส 1 (สำเร็จ) | เฟส 2 (กำลังดำเนินการ)", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[300], hAlign='CENTER')
    meta_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(PageBreak())
    
    # ── SECTION 1: EXECUTIVE SUMMARY ──────────────────────────────────────
    story.append(Paragraph("1. บทสรุปผู้บริหาร (Executive Summary)", h1_style))
    story.append(Paragraph(
        "โครงการ <b>Thai Omni-Modal AI (OMNI)</b> มีเป้าหมายในการพัฒนาแบบจำลองภาษาขนาดใหญ่ (LLM) ให้มีความสามารถระดับสากลในการรับรู้และตอบสนองหลายประสาทสัมผัส (Multi-Modal) สำหรับภาษาไทยโดยเฉพาะ โครงสร้างหลักประยุกต์ใช้โมเดลภาษา <b>Qwen3.5-0.8B</b> ประสานงานร่วมกับโมเดลถอดความเสียงระดับแนวหน้า <b>Typhoon-Whisper-Turbo</b> ของ SCB-10X โดยเชื่อมโยงผ่านโมดูลแปลงสัญญาณเสียง <b>AudioProjector</b> เพื่อเรียนรู้วิธีแปลงลักษณะของเสียงไทยเข้าไปยังระดับตัวแทนความรู้ (Hidden States) ของโมเดลภาษา",
        body_style
    ))
    story.append(Paragraph(
        "การพัฒนากำหนดขั้นตอนออกเป็น 4 เฟสหลัก เพื่อสร้างเสถียรภาพและคุณภาพสูงสุด ตั้งแต่การจัดตำแหน่งเสียง (Audio Alignment) การปรับแต่งคำสั่งแบบรวมประสาทสัมผัส (Omni SFT) การกลั่นกรององค์ความรู้ (Distillation) ไปจนถึงการส่งออกคำพูดเสียงไทยโดยตรง (Native Speech Output) รายงานฉบับนี้สรุปความคืบหน้าล่าสุดของการเอาชนะปัญหาคอขวดด้านข้อมูลเสียงและโครงสร้างทางสถาปัตยกรรมเพื่อเตรียมพร้อมสำหรับ SFT ระดับถัดไป",
        body_style
    ))
    
    # ── SECTION 2: SYSTEM ARCHITECTURE ───────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(Paragraph("2. สถาปัตยกรรมระบบ (Architecture Blueprint)", h1_style))
    story.append(Paragraph(
        "สถาปัตยกรรมของ OMNI ถูกออกแบบให้มีความยืดหยุ่นสูง รองรับทั้งการทำงานแบบปกติ (Dense) และแบบก้าวหน้า (Mixture-of-Experts) ดังรายละเอียดต่อไปนี้:",
        body_style
    ))
    
    story.append(Paragraph("2.1 โครงสร้างปกติ (Dense Architecture)", h2_style))
    story.append(Paragraph(
        "รับสัญญาณเสียง (.wav) ผ่านโมเดล Typhoon-Whisper-Turbo แปลงลักษณะเสียงผ่าน MLP AudioProjector เพื่อผสานเข้ากับโทเค็นข้อความดิบ และส่งเข้าไปประมวลผลต่อใน Qwen3.5-0.8B เพื่อผลิตผลลัพธ์ข้อความภาษาไทย หรือส่งต่อไปยัง AudioLMHead (EnCodec) สำหรับผลิตไฟล์เสียงพูดตอบกลับ",
        body_style
    ))

    story.append(Paragraph("2.2 โครงสร้างผู้เชี่ยวชาญแบบผสม (Sparse MoE Architecture)", h2_style))
    story.append(Paragraph(
        "เปลี่ยนชั้น FFN (Feed-Forward Network) ในโมเดลภาษาหลักให้เป็น <b>Sparse MoE Layer</b> โดยใช้ <b>Top-2 Gating Router</b> ในการจัดสรรให้โทเค็นวิ่งเข้าไปยังผู้เชี่ยวชาญ (Experts) 4 ด้านแยกขาดจากกันอย่างอิสระเพื่อประสิทธิภาพสูงสุด:",
        body_style
    ))
    
    # Expert Table
    expert_data = [
        [Paragraph("<b>ผู้เชี่ยวชาญ (Expert)</b>", body_style), Paragraph("<b>หน้าที่หลัก</b>", body_style), Paragraph("<b>แหล่งข้อมูลสำคัญ</b>", body_style)],
        [Paragraph("Expert 1: Audio", body_style), Paragraph("ตีความเสียง คลื่นเสียง ไวยากรณ์คำพูดไทย", body_style), Paragraph("Google FLEURS (th_th)", body_style)],
        [Paragraph("Expert 2: Vision/OCR", body_style), Paragraph("อ่านลายมือ สแกนภาพ โครงสร้างเอกสารไทย", body_style), Paragraph("LLaVA-Instruct-150K / ColPali", body_style)],
        [Paragraph("Expert 3: Text/Lang", body_style), Paragraph("ความเข้าใจหลักภาษา การสนทนาทั่วไป", body_style), Paragraph("FineTome-100k", body_style)],
        [Paragraph("Expert 4: Agent", body_style), Paragraph("การเรียกใช้เครื่องมือ คิดวิเคราะห์แบบ ReAct", body_style), Paragraph("xverse/agent-sft-data (ไทย)", body_style)]
    ]
    t_expert = Table(expert_data, colWidths=[110, 180, 200])
    t_expert.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_light),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_expert)
    
    story.append(PageBreak())
    
    # ── SECTION 3: DEVELOPMENT & TRAINING PHASES ─────────────────────────
    story.append(Paragraph("3. ขั้นตอนการพัฒนาและฝึกสอน (Training Phases)", h1_style))
    story.append(Paragraph(
        "กระบวนการฝึกสอนถูกจำแนกออกเป็น 4 เฟสหลัก เพื่อไม่ให้รบกวนความสามารถทางภาษาดั้งเดิมของโมเดล:",
        body_style
    ))
    
    phases_data = [
        [
            Paragraph("<b>เฟส (Phase)</b>", body_style),
            Paragraph("<b>ขอบเขตการเทรน / ปลดล็อกน้ำหนัก</b>", body_style),
            Paragraph("<b>เป้าหมายหลัก</b>", body_style)
        ],
        [
            Paragraph("Phase 1: Audio Alignment", body_style),
            Paragraph("🔥 AudioProjector MLP เท่านั้น<br/>❄️ ส่วนอื่นแช่แข็งทั้งหมด", body_style),
            Paragraph("แมปสัญญาณเสียงไทยเข้าหามิติคำศัพท์หลักของ LLM (1024-dim)", body_style)
        ],
        [
            Paragraph("Phase 2: Omni SFT", body_style),
            Paragraph("🔥 LLM Backbone + AudioProjector<br/>❄️ Audio/Vision Encoder แช่แข็ง", body_style),
            Paragraph("ฝึกฝนความเข้าใจแบบรวมประสาทสัมผัส โดยใช้ข้อมูลสัดส่วนผสม (Audio 35%, Text 35%, Vision 30%)", body_style)
        ],
        [
            Paragraph("Phase 3: Distillation", body_style),
            Paragraph("🔥 หัวใจระบบประมวลผลของโมเดลลูก", body_style),
            Paragraph("ส่งต่อโครงสร้างและความเร็วไปยังหัวประมวลผลเสียงพูดแบบ Streaming", body_style)
        ],
        [
            Paragraph("Phase 4: Native Speech Out", body_style),
            Paragraph("🔥 AudioLMHead เท่านั้น (~5.2M พารามิเตอร์)", body_style),
            Paragraph("แปลง Hidden States เป็นรหัสคำพูดตอบกลับเสียงสดไทยโดยตรง", body_style)
        ]
    ]
    t_phase = Table(phases_data, colWidths=[125, 175, 190])
    t_phase.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_light),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_phase)
    
    # ── SECTION 4: DATASET STRATEGY & TROUBLESHOOTING ─────────────────────
    story.append(Spacer(1, 15))
    story.append(Paragraph("4. การเลือกชุดข้อมูลและการปรับแก้ปัญหาเสถียรภาพ", h1_style))
    story.append(Paragraph(
        "ในระหว่างการดำเนินโครงการ ทีมงานได้เผชิญหน้าและก้าวผ่านปัญหาทางเทคนิคสำคัญหลายประการ:",
        body_style
    ))
    
    story.append(Paragraph("4.1 ปัญหาการฝังข้อมูลเสียงล้มเหลว (Training on Silence)", body_style))
    story.append(Paragraph(
        "<b>ปัญหา:</b> ชุดข้อมูลเริ่มต้น (Typhoon Audio Preview) มีเฉพาะข้อมูลระดับ Metadata (Path ไฟล์บน SCB-10X) แต่ไม่มีไฟล์คลื่นเสียงดิบ ส่งผลให้ระบบใช้ตัวแปรเสียงศูนย์ (Zero-tensor) และเทรนโมเดลบนความเงียบ รวมถึงมี NameError ในระบบกู้ภัยข้อผิดพลาด<br/>"
        "<b>แนวทางแก้ไข:</b> ย้ายระบบการจัดตำแหน่งเสียงมาใช้ชุดข้อมูล <b>Google FLEURS (th_th)</b> ซึ่งมีเสียงสดฝังอยู่ในระบบ Parquet ของ Hugging Face พร้อมแก้ไขโครงสร้างการดึงตัวแปรเสียงเพื่อลดข้อผิดพลาดทั้งหมด",
        bullet_style
    ))
    
    story.append(Paragraph("4.2 ปัญหา Dependency ของระบบเสียงและการหลีกเลี่ยง torchcodec", body_style))
    story.append(Paragraph(
        "<b>ปัญหา:</b> ไลบรารี <code>datasets</code> ของ Hugging Face มีการนำเข้า <code>torchcodec</code> แบบ Lazy ซึ่งจะกระตุ้นข้อผิดพลาด ImportError ใน DataLoader workers เครื่อง Cloud ที่ขาดไลบรารีระบบ FFmpeg<br/>"
        "<b>แนวทางแก้ไข:</b> สแกนคอลัมน์และแคสต์ประเภทคอลัมน์ที่เป็น Audio ทั้งหมดในขั้นตอนโหลดข้อมูลให้เป็น Plain Python Dictionary (มีโครงสร้าง <code>bytes</code> และ <code>path</code>) ถาวร จากนั้นทำการถอดรหัสเสียงดิบด้วยตนเองโดยใช้ไลบรารี <code>soundfile</code> และ <code>io.BytesIO</code> รวมถึงทำการจำลอง Downmix สเตอริโอเป็นโมโนเพื่อลดความคลาดเคลื่อนของขนาด",
        bullet_style
    ))

    story.append(Paragraph("4.3 ปัญหาโครงสร้างคอลัมน์ของชุดข้อมูลเสียงสนทนา (KeyError: 'response')", body_style))
    story.append(Paragraph(
        "<b>ปัญหา:</b> ชุดข้อมูล <code>typhoon-ai/chatbot-arena-spoken-voices</code> มีโครงสร้างคอลัมน์เสียงแตกต่างจากปกติ (เช่น <code>voice_a</code>, <code>voice_user</code>) และไม่มีคอลัมน์ <code>response</code> ส่งผลให้เกิด KeyError ในขั้นตอน SFT<br/>"
        "<b>แนวทางแก้ไข:</b> เพิ่มระบบตรวจสอบประเภทข้อมูลใน <code>__getitem__</code> ของ DataLoader เมื่อพบรูปแบบข้อมูลบทสนทนา จะสกัดเสียงพูดและทรานสคริปต์ของคำตอบผู้ช่วย (Assistant Response) จาก <code>conversation_a/voice_a</code> ออกมาใช้งานแบบพลวัตโดยอัตโนมัติ",
        bullet_style
    ))

    story.append(Paragraph("4.4 ข้อผิดพลาดชนิดข้อมูลของเลเยอร์ MoE (BFloat16 vs Float32)", body_style))
    story.append(Paragraph(
        "<b>ปัญหา:</b> เกิดข้อผิดพลาด <code>RuntimeError: index_add_(): self (BFloat16) and source (Float) must have the same scalar type</code> ในโมดูล <code>moe.py</code> เนื่องจากผลลัพธ์ของฟังก์ชัน Softmax จากเกตถูกแปลงเป็น Float32 เพื่อเสถียรภาพตัวเลข ในขณะที่ค่า Hidden States ของแบบจำลองอยู่ในโหมด BFloat16<br/>"
        "<b>แนวทางแก้ไข:</b> เพิ่มโค้ดแคสต์ชนิดข้อมูล <code>(expert_outputs * expert_weights).to(final_output.dtype)</code> ก่อนนำเข้าการประมวลผลคำสั่ง <code>index_add_</code> เพื่อรองรับความเข้ากันได้แบบ 100%",
        bullet_style
    ))

    story.append(Paragraph("4.5 การเตรียมความพร้อมการย้ายระบบประมวลผล (Session Switching Ready)", body_style))
    story.append(Paragraph(
        "พัฒนาและเขียนสคริปต์แบบรวมศูนย์ <code>scripts/start_phase2.py</code> เพื่อรองรับการทำงานย้ายข้ามเซสชัน GPU โดยอัตโนมัติ ซึ่งจะครอบคลุมขั้นตอนล็อกอิน ดาวน์โหลดน้ำหนัก ทดสอบคุณภาพเสียง (Inference Check) และสตาร์ท SFT ทันที",
        bullet_style
    ))

    # ── SECTION 5: ROADMAP ────────────────────────────────────────────────
    story.append(Spacer(1, 15))
    story.append(Paragraph("5. แผนการพัฒนาในอนาคต (Roadmap)", h1_style))
    
    roadmap_data = [
        [Paragraph("<b>เฟสปัจจุบัน</b>", body_style), Paragraph("ตรวจสอบเสถียรภาพโมเดลบน Phase 1 พร้อมอัปโหลดน้ำหนักตัวจัดวางเสียงไทย", body_style)],
        [Paragraph("<b>มิถุนายน 2026</b>", body_style), Paragraph("รัน Phase 2 SFT บน GPU Session ใหม่ด้วยความยาวข้อมูลรวมสูงสุด 1024 โทเค็น", body_style)],
        [Paragraph("<b>กรกฎาคม 2026</b>", body_style), Paragraph("แปลงโครงสร้าง Qwen3.5 เป็น 4-Expert MoE โดยการแช่แข็งความรู้ปกติและปรับแต่ง Router", body_style)],
        [Paragraph("<b>สิงหาคม 2026</b>", body_style), Paragraph("ฝึกสอนขั้นสุดท้ายสำหรับหัวส่งออกเสียงภาษาไทยสด (Native Speech Streaming out)", body_style)]
    ]
    t_roadmap = Table(roadmap_data, colWidths=[100, 390])
    t_roadmap.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), c_light),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_roadmap)
    
    # ── BUILD DOCUMENT ────────────────────────────────────────────────────
    doc.build(story)
    print(f"Report successfully generated at: {output_filename}")


if __name__ == "__main__":
    make_report()
