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


def snap_rotation_to_allowed(rotation: int, allowed_rotations: list[int]) -> int:
    """Pick the closest rotation within ``allowed_rotations``.

    The distance is measured cyclically (e.g. 350 is close to 0).
    """

    def _distance(a: int, b: int) -> int:
        return min((a - b) % 360, (b - a) % 360)

    return min(allowed_rotations, key=lambda allowed: _distance(rotation % 360, allowed % 360))


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

        # まず縦横比を確認し、縦長なら0/180、横長なら90/270の範囲に絞る
        is_portrait = img.height >= img.width
        allowed_rotations = [0, 180] if is_portrait else [90, 270]

        # 信頼度が低いページは、これまでの高信頼ページの多数決で補完する
        used_fallback = False
        if conf < CONF_THRESHOLD and high_conf_rots:
            applied_rot = Counter(high_conf_rots).most_common(1)[0][0]
            used_fallback = True
        else:
            applied_rot = rot % 360

        snapped_rot = snap_rotation_to_allowed(applied_rot, allowed_rotations)
        if conf >= CONF_THRESHOLD:
            high_conf_rots.append(snapped_rot)

        # rot(0/90/180/270)を、そのページの回転として設定
        # もし向きが逆に出る場合は、(360-rot)%360 に変えてください
        if snapped_rot % 360 != 0:
            page.Rotate = snapped_rot
            changed += 1

        if conf < CONF_THRESHOLD:
            low.append((i, rot, conf, used_fallback, snapped_rot))

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
