"""Build polished PDF editions of the Pluto RTSA/VSA/VSG Markdown manuals."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs" / "user-manual"
OUTPUT_DIR = ROOT / "output" / "pdf"
FONT_REGULAR = Path(r"C:\Windows\Fonts\meiryo.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\meiryob.ttc")
PAGE_SIZE = A4
PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE
MARGIN_X = 18 * mm
MARGIN_TOP = 17 * mm
MARGIN_BOTTOM = 16 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X

MANUALS = (
    ("Pluto_RTSA_User_Manual_JA.md", "Pluto_RTSA_User_Manual_JA.pdf", "Spectrum Analyzer"),
    ("Pluto_VSA_User_Manual_JA.md", "Pluto_VSA_User_Manual_JA.pdf", "Vector Signal Analyzer"),
    ("Pluto_VSG_User_Manual_JA.md", "Pluto_VSG_User_Manual_JA.pdf", "Vector Signal Generator"),
)


def _register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("ManualJP", str(FONT_REGULAR), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("ManualJP-Bold", str(FONT_BOLD), subfontIndex=0))
    pdfmetrics.registerFontFamily(
        "ManualJP",
        normal="ManualJP",
        bold="ManualJP-Bold",
        italic="ManualJP",
        boldItalic="ManualJP-Bold",
    )


class ManualDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, *, title: str) -> None:
        super().__init__(
            filename,
            pagesize=PAGE_SIZE,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title=title,
            author="Pluto Spectrum App Project",
        )
        self.manual_title = title
        frame = Frame(
            MARGIN_X,
            MARGIN_BOTTOM,
            CONTENT_WIDTH,
            PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate("manual", [frame], onPage=self._draw_chrome))
        self._bookmark_index = 0

    def beforeDocument(self) -> None:
        self._bookmark_index = 0

    def _draw_chrome(self, canvas, doc) -> None:
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#666666"))
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN_X, PAGE_HEIGHT - 11 * mm, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 11 * mm)
        canvas.setFont("ManualJP", 8)
        canvas.setFillColor(colors.HexColor("#444444"))
        canvas.drawString(MARGIN_X, PAGE_HEIGHT - 8 * mm, self.manual_title)
        canvas.drawRightString(PAGE_WIDTH - MARGIN_X, 8 * mm, f"{doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        level = getattr(flowable, "_toc_level", None)
        if level is None:
            return
        self._bookmark_index += 1
        key = f"heading-{self._bookmark_index}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(flowable.getPlainText(), key, level=level)
        self.notify("TOCEntry", (level, flowable.getPlainText(), self.page, key))


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyJP",
        parent=base["BodyText"],
        fontName="ManualJP",
        fontSize=9.3,
        leading=13.6,
        textColor=colors.HexColor("#222222"),
        spaceAfter=4,
        wordWrap="CJK",
    )
    return {
        "body": body,
        "cover": ParagraphStyle(
            "Cover",
            parent=body,
            fontName="ManualJP-Bold",
            fontSize=28,
            leading=38,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#143a52"),
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=body,
            fontSize=14,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#52636e"),
        ),
        "h2": ParagraphStyle(
            "Heading2JP",
            parent=body,
            fontName="ManualJP-Bold",
            fontSize=17,
            leading=23,
            textColor=colors.HexColor("#0b607d"),
            spaceAfter=9,
            borderWidth=0,
            borderPadding=(0, 0, 4, 0),
        ),
        "h3": ParagraphStyle(
            "Heading3JP",
            parent=body,
            fontName="ManualJP-Bold",
            fontSize=12,
            leading=17,
            textColor=colors.HexColor("#174f64"),
            spaceBefore=6,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "BulletJP",
            parent=body,
            leftIndent=14,
            firstLineIndent=-8,
            bulletIndent=3,
            spaceAfter=2,
        ),
        "quote": ParagraphStyle(
            "QuoteJP",
            parent=body,
            leftIndent=12,
            rightIndent=12,
            borderColor=colors.HexColor("#d89b28"),
            borderWidth=1,
            borderPadding=7,
            backColor=colors.HexColor("#fff7df"),
        ),
        "code": ParagraphStyle(
            "CodeJP",
            parent=body,
            fontName="Courier",
            fontSize=8,
            leading=11,
            leftIndent=8,
            borderColor=colors.HexColor("#c5cbd0"),
            borderWidth=0.5,
            borderPadding=6,
            backColor=colors.HexColor("#f4f6f7"),
        ),
        "caption": ParagraphStyle(
            "CaptionJP",
            parent=body,
            fontSize=8,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
        ),
        "toc_title": ParagraphStyle(
            "TOCTitleJP",
            parent=body,
            fontName="ManualJP-Bold",
            fontSize=20,
            leading=26,
            textColor=colors.HexColor("#0b607d"),
        ),
    }


def _inline(text: str) -> str:
    value = html.escape(text.strip())
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<font color="#006c91">\1</font>', value)
    value = value.replace("  ", " ")
    return value


def _table(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    column_count = max(len(row) for row in rows)
    data = []
    for row_index, row in enumerate(rows):
        style = styles["h3"] if row_index == 0 else styles["body"]
        data.append([Paragraph(_inline(cell), style) for cell in row + [""] * (column_count - len(row))])
    widths = [CONTENT_WIDTH / column_count] * column_count
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24576b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "ManualJP-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7f9fa")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#aeb8bd")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _parse_table_line(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _image_flowable(source: Path, alt: str, styles: dict[str, ParagraphStyle]):
    image = Image(str(source))
    max_width = CONTENT_WIDTH
    max_height = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM - 22 * mm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return KeepTogether([image, Spacer(1, 2 * mm), Paragraph(_inline(alt), styles["caption"])])


def _markdown_story(path: Path, subtitle: str, styles: dict[str, ParagraphStyle]):
    lines = path.read_text(encoding="utf-8").splitlines()
    title = next(line[2:].strip() for line in lines if line.startswith("# "))
    story = [
        Spacer(1, 42 * mm),
        Paragraph(_inline(title), styles["cover"]),
        Spacer(1, 8 * mm),
        Paragraph(_inline(subtitle), styles["subtitle"]),
        Spacer(1, 65 * mm),
        Paragraph("Pluto Spectrum App Project", styles["subtitle"]),
        PageBreak(),
        Paragraph("目次", styles["toc_title"]),
        Spacer(1, 6 * mm),
    ]
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC1", fontName="ManualJP", fontSize=10, leading=15, leftIndent=0),
        ParagraphStyle("TOC2", fontName="ManualJP", fontSize=8.5, leading=12, leftIndent=16),
    ]
    story.extend([toc, PageBreak()])

    index = 0
    first_section = True
    while index < len(lines):
        line = lines[index].rstrip()
        if line.startswith("# ") or line.startswith("文書版:") or line.startswith("対象:"):
            index += 1
            continue
        if not line:
            index += 1
            continue
        if line.startswith("## "):
            if not first_section:
                story.append(PageBreak())
            first_section = False
            paragraph = Paragraph(_inline(line[3:]), styles["h2"])
            paragraph._toc_level = 0
            story.append(paragraph)
            index += 1
            continue
        if line.startswith("### "):
            paragraph = Paragraph(_inline(line[4:]), styles["h3"])
            paragraph._toc_level = 1
            story.append(paragraph)
            index += 1
            continue
        image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
        if image_match:
            image_path = (path.parent / image_match.group(2)).resolve()
            story.append(_image_flowable(image_path, image_match.group(1), styles))
            index += 1
            continue
        if line.startswith("```"):
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index])
                index += 1
            story.append(Preformatted("\n".join(code_lines), styles["code"]))
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                row = _parse_table_line(lines[index])
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in row):
                    rows.append(row)
                index += 1
            story.extend([_table(rows, styles), Spacer(1, 3 * mm)])
            continue
        if line.startswith("> "):
            story.append(Paragraph(_inline(line[2:]), styles["quote"]))
            index += 1
            continue
        bullet = re.match(r"^(-|\d+\.)\s+(.*)$", line)
        if bullet:
            marker = "・" if bullet.group(1) == "-" else bullet.group(1)
            story.append(Paragraph(_inline(f"{marker} {bullet.group(2)}"), styles["bullet"]))
            index += 1
            continue

        paragraph_lines = [line]
        index += 1
        while index < len(lines):
            next_line = lines[index].rstrip()
            if (
                not next_line
                or next_line.startswith(("#", "|", ">", "```", "!["))
                or re.match(r"^(-|\d+\.)\s+", next_line)
            ):
                break
            paragraph_lines.append(next_line)
            index += 1
        story.append(Paragraph(_inline(" ".join(paragraph_lines)), styles["body"]))
    return title, story


def build_all() -> list[Path]:
    _register_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    outputs = []
    for source_name, output_name, subtitle in MANUALS:
        source = SOURCE_DIR / source_name
        output = OUTPUT_DIR / output_name
        title, story = _markdown_story(source, subtitle, styles)
        document = ManualDocTemplate(str(output), title=title)
        document.multiBuild(story)
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    for built in build_all():
        print(built)
