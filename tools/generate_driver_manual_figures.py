"""Generate stable, hardware-neutral figures for the Pluto driver guide."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "images" / "user-manual"
FONT_REGULAR = Path(r"C:\Windows\Fonts\meiryo.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\meiryob.ttc")

NAVY = "#123B52"
TEAL = "#08799A"
CYAN = "#DDF4FA"
GREEN = "#27A86B"
GREEN_BG = "#E4F5EC"
ORANGE = "#E28A2B"
ORANGE_BG = "#FFF0DC"
GRAY = "#5D6B75"
LIGHT = "#F3F6F8"
LINE = "#AAB7BF"
WHITE = "#FFFFFF"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


def _canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1600, 900), WHITE)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1600, 118), fill=NAVY)
    draw.text((70, 25), title, font=_font(42, bold=True), fill=WHITE)
    draw.text((72, 78), subtitle, font=_font(21), fill="#CFE4EF")
    return image, draw


def _centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, *,
              font: ImageFont.FreeTypeFont, fill: str, spacing: int = 8) -> None:
    bounds = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x1, y1, x2, y2 = box
    draw.multiline_text(
        ((x1 + x2 - width) / 2, (y1 + y2 - height) / 2),
        text,
        font=font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = TEAL) -> None:
    draw.line((start, end), fill=color, width=9)
    x, y = end
    draw.polygon(((x, y), (x - 24, y - 15), (x - 24, y + 15)), fill=color)


def build_install_flow() -> None:
    image, draw = _canvas(
        "Windowsドライバ導入の流れ",
        "接続順序を守ることで、複合USBデバイスの登録失敗を避けます",
    )
    steps = [
        ("1", "PlutoをPCから外す", "ドライバ導入前は未接続"),
        ("2", "公式インストーラを実行", "Analog Devices配布元を確認"),
        ("3", "Windowsを再起動", "要求された場合は必ず実施"),
        ("4", "USB端子へ接続", "データ通信対応ケーブルを使用"),
    ]
    box_width = 330
    gap = 45
    left = 72
    top = 250
    for index, (number, heading, note) in enumerate(steps):
        x1 = left + index * (box_width + gap)
        x2 = x1 + box_width
        draw.rounded_rectangle((x1, top, x2, top + 330), radius=22, fill=LIGHT, outline=LINE, width=3)
        draw.ellipse((x1 + 119, top - 54, x1 + 211, top + 38), fill=TEAL)
        _centered(draw, (x1 + 119, top - 54, x1 + 211, top + 38), number, font=_font(38, bold=True), fill=WHITE)
        _centered(draw, (x1 + 24, top + 70, x2 - 24, top + 175), heading, font=_font(28, bold=True), fill=NAVY)
        draw.line((x1 + 38, top + 205, x2 - 38, top + 205), fill=LINE, width=2)
        _centered(draw, (x1 + 30, top + 220, x2 - 30, top + 300), note, font=_font(20), fill=GRAY)
        if index < len(steps) - 1:
            _arrow(draw, (x2 + 7, top + 165), (x2 + gap - 8, top + 165))
    draw.rounded_rectangle((230, 690, 1370, 790), radius=18, fill=ORANGE_BG, outline=ORANGE, width=3)
    _centered(
        draw,
        (260, 705, 1340, 775),
        "インストール完了前にPlutoを接続しない",
        font=_font(27, bold=True),
        fill="#8B4C10",
    )
    image.save(OUTPUT_DIR / "pluto-driver-install-flow.png", optimize=True)


def build_usb_ports() -> None:
    image, draw = _canvas(
        "USB端子とケーブルの選び方",
        "通常のPC接続には、Pluto本体の「USB」表記側を使用します",
    )
    draw.rounded_rectangle((560, 220, 1240, 700), radius=28, fill="#E7EEF2", outline=NAVY, width=5)
    draw.text((760, 310), "ADALM-PLUTO", font=_font(40, bold=True), fill=NAVY)
    draw.text((796, 375), "SDR", font=_font(30), fill=GRAY)

    # Stylized connector edge. The labels, not physical orientation, are authoritative.
    draw.rounded_rectangle((520, 340, 610, 445), radius=10, fill="#2B343A")
    draw.rounded_rectangle((520, 535, 610, 640), radius=10, fill="#2B343A")
    draw.text((625, 360), "USB", font=_font(28, bold=True), fill=TEAL)
    draw.text((625, 555), "PWR", font=_font(28, bold=True), fill=ORANGE)

    draw.rounded_rectangle((80, 250, 390, 430), radius=22, fill=GREEN_BG, outline=GREEN, width=4)
    _centered(draw, (100, 270, 370, 345), "Windows PC", font=_font(33, bold=True), fill=NAVY)
    _centered(draw, (100, 345, 370, 410), "データ通信 + 給電", font=_font(21), fill=GREEN)
    _arrow(draw, (390, 340), (520, 392), GREEN)
    draw.text((372, 282), "データ対応USBケーブル", font=_font(20, bold=True), fill=GREEN)

    draw.rounded_rectangle((80, 540, 390, 720), radius=22, fill=ORANGE_BG, outline=ORANGE, width=4)
    _centered(draw, (100, 560, 370, 635), "外部5 V電源", font=_font(30, bold=True), fill=NAVY)
    _centered(draw, (100, 635, 370, 700), "給電専用（必要時）", font=_font(21), fill="#8B4C10")
    _arrow(draw, (390, 630), (520, 588), ORANGE)

    draw.rounded_rectangle((1265, 275, 1530, 650), radius=22, fill=LIGHT, outline=LINE, width=3)
    draw.text((1300, 310), "確認ポイント", font=_font(28, bold=True), fill=NAVY)
    checks = ["USB表記側", "データ対応", "短いケーブル", "PC本体へ直結"]
    for index, label in enumerate(checks):
        y = 390 + index * 58
        draw.ellipse((1300, y, 1332, y + 32), fill=GREEN)
        draw.line((1308, y + 17, 1317, y + 26), fill=WHITE, width=4)
        draw.line((1317, y + 26, 1328, y + 8), fill=WHITE, width=4)
        draw.text((1350, y - 1), label, font=_font(20), fill=GRAY)
    image.save(OUTPUT_DIR / "pluto-driver-usb-ports.png", optimize=True)


def build_recognition_check() -> None:
    image, draw = _canvas(
        "正常認識の確認ポイント",
        "Plutoは複数機能を持つUSBデバイスとしてWindowsへ登録されます",
    )
    labels = [
        ("IIO USB device", "アプリからIQを送受信"),
        ("Serial (COM)", "保守・コンソール接続"),
        ("Mass Storage", "PlutoSDRドライブ"),
        ("USB Ethernet", "192.168.2.1への接続"),
    ]
    for index, (heading, note) in enumerate(labels):
        y = 210 + index * 125
        draw.rounded_rectangle((95, y, 750, y + 90), radius=16, fill=LIGHT, outline=LINE, width=3)
        draw.ellipse((125, y + 24, 168, y + 67), fill=GREEN)
        draw.line((136, y + 45, 148, y + 57), fill=WHITE, width=5)
        draw.line((148, y + 57, 161, y + 34), fill=WHITE, width=5)
        draw.text((195, y + 14), heading, font=_font(25, bold=True), fill=NAVY)
        draw.text((195, y + 51), note, font=_font(18), fill=GRAY)

    _arrow(draw, (785, 435), (930, 435))
    draw.rounded_rectangle((960, 245, 1505, 635), radius=24, fill=CYAN, outline=TEAL, width=4)
    draw.text((1075, 290), "Pluto RTSA / VSA / VSG", font=_font(27, bold=True), fill=NAVY)
    draw.rounded_rectangle((1030, 380, 1435, 475), radius=12, fill=WHITE, outline=LINE, width=3)
    draw.text((1060, 398), "Device / Instrument", font=_font(20), fill=GRAY)
    draw.text((1060, 435), "serial:xxxxxxxx", font=_font(24, bold=True), fill=TEAL)
    draw.rounded_rectangle((1110, 525, 1360, 590), radius=12, fill=TEAL)
    _centered(draw, (1110, 525, 1360, 590), "Refresh", font=_font(23, bold=True), fill=WHITE)
    draw.rounded_rectangle((290, 755, 1310, 830), radius=16, fill=GREEN_BG, outline=GREEN, width=3)
    _centered(
        draw,
        (315, 765, 1285, 820),
        "黄色い警告アイコンや「不明なデバイス」がなければ正常です",
        font=_font(24, bold=True),
        fill="#176C47",
    )
    image.save(OUTPUT_DIR / "pluto-driver-recognition-check.png", optimize=True)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_install_flow()
    build_usb_ports()
    build_recognition_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
