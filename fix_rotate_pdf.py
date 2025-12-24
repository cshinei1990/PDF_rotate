from pathlib import Path
from collections import Counter

import pikepdf
from pdf2image import convert_from_path
import pytesseract
import tkinter as tk
from tkinter import filedialog

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


def determine_output_path(inp: Path) -> Path:
    """Return output path with ``_rot`` suffix in the same directory.

    If a file with that name already exists, append a sequential index
    (``_1``, ``_2``, ...) to avoid overwriting existing outputs.
    """

    base = inp.with_name(f"{inp.stem}_rot{inp.suffix}")
    if not base.exists():
        return base

    for idx in range(1, 1_000):
        candidate = inp.with_name(f"{inp.stem}_rot_{idx}{inp.suffix}")
        if not candidate.exists():
            return candidate

    raise FileExistsError("適切な出力ファイル名を決定できませんでした。既存の rot ファイルを整理してください。")


def save_pdf(pdf: pikepdf.Pdf, out: Path) -> Path:
    """Save ``pdf`` to ``out`` while handling Windows permission quirks."""

    out.parent.mkdir(parents=True, exist_ok=True)

    def _try_save(path: Path):
        if path.exists():
            try:
                path.unlink()
            except PermissionError:
                # File may be open in another app; surface a clear message.
                raise PermissionError(
                    f"出力ファイル {path} を上書きできません。閲覧アプリを閉じて再実行してください。"
                )
        pdf.save(str(path))

    try:
        _try_save(out)
    except PermissionError as exc:
        # As a fallback, try a slightly different name to avoid failure on locked files.
        alt = out.with_name(f"{out.stem}_new{out.suffix}")
        print(f"{exc}\n{out} に保存できなかったため {alt} に書き込みます。")
        _try_save(alt)
        out = alt
    finally:
        pdf.close()

    return out


def process_file(inp: Path) -> None:
    out = determine_output_path(inp)

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

    out = save_pdf(pdf, out)

    print(f"Saved: {out}  changed_pages={changed}")
    if low:
        print("Low-confidence pages (please verify):")
        for p, r, c, used_fallback, applied in low:
            suffix = " (fallback used)" if used_fallback else ""
            print(f"  page {p}: rot={applied}, conf={c}{suffix}")


if __name__ == "__main__":
    import sys

    # コマンドライン引数があれば従来通り使う。なければファイルダイアログで選択。
    if len(sys.argv) >= 2:
        input_files = sys.argv[1:]
    else:
        root = tk.Tk()
        root.withdraw()
        selected_files = filedialog.askopenfilenames(
            title="回転を補正するPDFを選択してください",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        input_files = list(selected_files)

        if not input_files:
            print("入力ファイルが選択されなかったため終了します。")
            raise SystemExit(1)

    for input_file in input_files:
        process_file(Path(input_file))
