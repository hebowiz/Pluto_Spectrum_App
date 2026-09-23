"""Build polished PDF editions of the Pluto application and setup manuals."""

from __future__ import annotations

import html
import argparse
import math
import os
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
    CondPageBreak,
    Frame,
    Flowable,
    Image,
    KeepTogether,
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
    (
        "Pluto_VSA_Analysis_Guide_JA.md",
        "Pluto_VSA_Analysis_Guide_JA.pdf",
        "Analysis Flow and Algorithms",
    ),
)

PDF_METADATA_TITLES = {
    "Pluto_RTSA_User_Manual_JA.md": "Pluto RTSA User Manual",
    "Pluto_VSA_User_Manual_JA.md": "Pluto VSA User Manual",
    "Pluto_VSG_User_Manual_JA.md": "Pluto VSG User Manual",
    "Pluto_VSA_Analysis_Guide_JA.md": "Pluto VSA Analysis Flow and Algorithms",
    "Pluto_Driver_Installation_Guide_JA.md": "ADALM-Pluto Windows Driver Installation Guide",
}


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
    def __init__(self, filename: str, *, title: str, metadata_title: str | None = None) -> None:
        super().__init__(
            filename,
            pagesize=PAGE_SIZE,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title=metadata_title or title,
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
        if level == 0:
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
            fontSize=22,
            leading=30,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#143a52"),
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=body,
            fontSize=11,
            leading=17,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#52636e"),
        ),
        "h2": ParagraphStyle(
            "Heading2JP",
            parent=body,
            fontName="ManualJP-Bold",
            fontSize=17,
            leading=23,
            textColor=colors.HexColor("#0b607d"),
            spaceBefore=12,
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
            fontName="ManualJP",
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
        "table_header": ParagraphStyle(
            "TableHeaderJP", parent=body, fontName="ManualJP-Bold",
            textColor=colors.white, fontSize=9.3, leading=13.6,
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
    value = re.sub(r"`([^`]+)`", r'<font name="ManualJP">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    def link(match):
        target = html.unescape(match.group(2))
        if not re.match(r"https?://|mailto:|#", target):
            pdf_names = {source: output for source, output, _ in MANUALS}
            target = pdf_names.get(target) or Path(os.path.relpath(SOURCE_DIR / target, OUTPUT_DIR)).as_posix()
        return f'<link href="{html.escape(target, quote=True)}" color="#006c91">{match.group(1)}</link>'
    value = re.sub(r"\[([^]]+)\]\(([^)]+)\)", link, value)
    value = value.replace("  ", " ")
    return value


def _table(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    column_count = max(len(row) for row in rows)
    data = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if row_index == 0 else styles["body"]
        data.append([Paragraph(_inline(cell), style) for cell in row + [""] * (column_count - len(row))])
    widths = [CONTENT_WIDTH / column_count] * column_count
    if column_count == 2:
        widths = [CONTENT_WIDTH * 0.34, CONTENT_WIDTH * 0.66]
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
    max_height = 155 * mm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return KeepTogether([image, Spacer(1, 2 * mm), Paragraph(_inline(alt), styles["caption"])])


class AnalysisFlowchart(Flowable):
    """Render the manuals' small Mermaid DAGs as indivisible PDF vectors.

    Supports named rectangular/decision nodes and labelled directed edges.
    Both source directions are arranged top-to-bottom to fit a portrait page.
    Unsupported syntax fails explicitly instead of silently omitting content.
    """

    def __init__(self, lines: list[str], styles: dict[str, ParagraphStyle]) -> None:
        super().__init__()
        self.width = CONTENT_WIDTH
        self.spaceBefore = 6
        self.spaceAfter = 9
        labels = {}
        decisions = set()
        self.edges = []
        node_pattern = re.compile(r"([A-Za-z]\w*)(?:\[([^\]]+)\]|\{([^}]+)\})?")

        def node(value):
            match = node_pattern.fullmatch(value.strip())
            if match is None:
                raise ValueError(f"Unsupported flowchart node: {value}")
            key, rectangle, decision = match.groups()
            if rectangle or decision:
                labels[key] = rectangle or decision
            else:
                labels.setdefault(key, key)
            if decision:
                decisions.add(key)
            return key

        for line in lines:
            line = line.strip()
            if not line or line in ("flowchart TD", "flowchart LR"):
                continue
            edge = re.fullmatch(r"(.+?)\s*-->\s*(?:\|([^|]+)\|\s*)?(.+)", line)
            if edge is None:
                raise ValueError(f"Unsupported flowchart statement: {line}")
            self.edges.append((node(edge[1]), node(edge[3]), edge[2] or ""))
        levels = {}
        while len(levels) < len(labels):
            progress = False
            for key in labels:
                parents = [a for a, b, _ in self.edges if b == key]
                if key not in levels and all(p in levels for p in parents):
                    levels[key] = max((levels[p] + 1 for p in parents), default=0)
                    progress = True
            if not progress:
                raise ValueError("Flowchart must be acyclic")
        node_style = ParagraphStyle(
            "FlowNode", parent=styles["body"], fontSize=9, leading=12,
            alignment=TA_CENTER, spaceAfter=0,
        )
        self.nodes = {}
        order_positions = {}
        y = 0
        for level in range(max(levels.values()) + 1):
            keys = [key for key in labels if levels[key] == level]
            def parent_position(key):
                parents = [a for a, b, _ in self.edges if b == key]
                return sum(order_positions[p] for p in parents) / len(parents) if parents else 0.5
            keys.sort(key=parent_position)
            for i, key in enumerate(keys):
                order_positions[key] = (i + 0.5) / len(keys)
            usable = self.width - 44
            width = min(320, (usable - 14 * (len(keys) - 1)) / len(keys))
            paragraphs = [Paragraph(_inline(labels[key]), node_style) for key in keys]
            heights = [p.wrap(width - 16, 1000)[1] for p in paragraphs]
            height = max(heights) + 16
            left = (self.width - len(keys) * width - (len(keys) - 1) * 14) / 2
            for i, (key, paragraph, text_height) in enumerate(zip(keys, paragraphs, heights)):
                self.nodes[key] = [left + i * (width + 14), y, width, height, paragraph, text_height, key in decisions]
            y += height + 28
        self.height = y - 28
        for item in self.nodes.values():
            item[1] = self.height - item[1] - item[3]
        self.levels = levels
        if self.height > PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM:
            raise ValueError("Flowchart exceeds one page")

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#52788a"))
        canvas.setLineWidth(0.8)
        for source, target, label in self.edges:
            sx, sy, sw, sh, *_ = self.nodes[source]
            tx, ty, tw, th, *_ = self.nodes[target]
            if self.levels[target] == self.levels[source] + 1:
                points = [(sx + sw / 2, sy), (tx + tw / 2, ty + th)]
            else:
                right = tx + tw / 2 >= self.width / 2
                lane = self.width - 7 if right else 7
                points = [(sx + (sw if right else 0), sy + sh / 2),
                          (lane, sy + sh / 2), (lane, ty + th / 2),
                          (tx + (tw if right else 0), ty + th / 2)]
            path = canvas.beginPath()
            path.moveTo(*points[0])
            for point in points[1:]:
                path.lineTo(*point)
            canvas.drawPath(path)
            x, y = points[-1]
            px, py = points[-2]
            angle = math.atan2(y - py, x - px)
            arrow = canvas.beginPath()
            arrow.moveTo(x, y)
            for offset in (-0.5, 0.5):
                arrow.lineTo(x - 5 * math.cos(angle + offset), y - 5 * math.sin(angle + offset))
            arrow.close()
            canvas.setFillColor(colors.HexColor("#52788a"))
            canvas.drawPath(arrow, fill=1, stroke=0)
            if label:
                canvas.setFont("ManualJP", 8)
                canvas.setFillColor(colors.HexColor("#174f64"))
                canvas.drawCentredString((points[0][0] + x) / 2, (points[0][1] + y) / 2 + 3, label)
        for x, y, width, height, paragraph, text_height, decision in self.nodes.values():
            canvas.setFillColor(colors.HexColor("#fff4d6" if decision else "#eef5f8"))
            canvas.roundRect(x, y, width, height, 5, fill=1, stroke=1)
            paragraph.drawOn(canvas, x + 8, y + (height - text_height) / 2)
        canvas.restoreState()


def _markdown_story(path: Path, subtitle: str, styles: dict[str, ParagraphStyle]):
    lines = path.read_text(encoding="utf-8").splitlines()
    title = next(line[2:].strip() for line in lines if line.startswith("# "))
    story = [
        Paragraph(_inline(title), styles["cover"]),
        Paragraph(_inline(subtitle), styles["subtitle"]),
    ]
    for line in lines:
        if line.startswith(("文書版:", "対象:", "アプリ仕様の確認基準:")):
            story.append(Paragraph(_inline(line), styles["body"]))
    story.extend([
        Spacer(1, 4 * mm),
        Paragraph("目次", styles["toc_title"]),
        Spacer(1, 2 * mm),
    ])
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC1", fontName="ManualJP", fontSize=10, leading=15, leftIndent=0),
        ParagraphStyle("TOC2", fontName="ManualJP", fontSize=8.5, leading=12, leftIndent=16),
    ]
    story.extend([toc, Spacer(1, 3 * mm)])

    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if line.startswith(("# ", "文書版:", "対象:", "アプリ仕様の確認基準:")):
            index += 1
            continue
        if not line:
            index += 1
            continue
        if line.startswith("## "):
            paragraph = Paragraph(_inline(line[3:]), styles["h2"])
            paragraph._toc_level = 0
            story.extend([CondPageBreak(65), paragraph])
            index += 1
            continue
        if line.startswith("### "):
            paragraph = Paragraph(_inline(line[4:]), styles["h3"])
            paragraph._toc_level = 1
            story.extend([CondPageBreak(50), paragraph])
            index += 1
            continue
        image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
        if image_match:
            image_path = (path.parent / image_match.group(2)).resolve()
            story.append(_image_flowable(image_path, image_match.group(1), styles))
            index += 1
            continue
        if line.startswith("```"):
            language = line[3:].strip()
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index])
                index += 1
            if language == "mermaid":
                story.append(AnalysisFlowchart(code_lines, styles))
            else:
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


def build_all(*, include_driver: bool = False) -> list[Path]:
    _register_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    outputs = []
    manuals = MANUALS
    if include_driver:
        manuals += (("Pluto_Driver_Installation_Guide_JA.md", "Pluto_Driver_Installation_Guide_JA.pdf", "Windows USB Driver Setup"),)
    for source_name, output_name, subtitle in manuals:
        source = SOURCE_DIR / source_name
        output = OUTPUT_DIR / output_name
        title, story = _markdown_story(source, subtitle, styles)
        document = ManualDocTemplate(
            str(output),
            title=title,
            metadata_title=PDF_METADATA_TITLES.get(source_name),
        )
        document.multiBuild(story)
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-driver", action="store_true")
    args = parser.parse_args()
    for built in build_all(include_driver=args.include_driver):
        print(built)
