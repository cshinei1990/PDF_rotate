from pathlib import Path
from collections import Counter

import pikepdf
from pdf2image import convert_from_path
import pytesseract

CONF_THRESHOLD = 5.0   # 漫画向けに低め。まずはこのくらいから
DPI = 200              # 低すぎるとOSD精度が落ちやすい

def detect_rotation_osd(pil_img):
    try:
        # Use dict output to avoid regex parsing and convert to RGB to ensure DPI metadata
        osd = pytesseract.image_to_osd(
            pil_img.convert("RGB"), output_type=pytesseract.Output.DICT
        )
        rot = int(osd.get("rotate", 0))
        conf = float(osd.get("orientation_confidence", 0.0))
    except pytesseract.TesseractError as exc:
        # When Tesseract fails due to few characters or missing resolution, treat as unknown
        print(f"Tesseract OSD failed ({exc}); using default rotation=0, conf=0")
        rot, conf = 0, 0.0

    return rot, conf

def main(inp, out):
    inp = Path(inp)
    out = Path(out)

    images = convert_from_path(str(inp), dpi=DPI)
    pdf = pikepdf.open(str(inp))

    low = []
    changed = 0
    high_conf_rots = []

    for i, (img, page) in enumerate(zip(images, pdf.pages), start=1):
        rot, conf = detect_rotation_osd(img)

        # 信頼度が低いページは、これまでの高信頼ページの多数決で補完する
        applied_rot = rot
        used_fallback = False
        if conf < CONF_THRESHOLD and high_conf_rots:
            applied_rot = Counter(high_conf_rots).most_common(1)[0][0]
            used_fallback = True
        else:
            high_conf_rots.append(rot % 360)

        # rot(0/90/180/270)を、そのページの回転として設定
        # もし向きが逆に出る場合は、(360-rot)%360 に変えてください
        if applied_rot % 360 != 0:
            page.Rotate = applied_rot
            changed += 1

        if conf < CONF_THRESHOLD:
            low.append((i, rot, conf, used_fallback, applied_rot))

    pdf.save(str(out))
    pdf.close()

    print(f"Saved: {out}  changed_pages={changed}")
    if low:
        print("Low-confidence pages (please verify):")
        for p, r, c, used_fallback, applied in low:
            suffix = " (fallback used)" if used_fallback else ""
            print(f"  page {p}: rot={applied}, conf={c}{suffix}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python fix_rotate_pdf.py input.pdf output.pdf")
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2])
