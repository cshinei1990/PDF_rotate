from pathlib import Path
from collections import Counter

import numpy as np
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


def page_contains_text(pil_img) -> bool:
    try:
        data = pytesseract.image_to_data(pil_img.convert("RGB"), output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        return False

    text_entries = data.get("text", [])
    return any(entry.strip() for entry in text_entries)


def estimate_updown_with_pose(pil_img):
    """Estimate upright orientation (0 or 180) using a pose estimation model.

    Returns (rotation, confidence). If no pose can be detected or mediapipe is not
    installed, returns (None, 0.0).
    """

    try:
        import mediapipe as mp
    except ImportError:
        print("mediapipe が見つからないためポーズ推定をスキップします。pip install mediapipe で追加できます。")
        return None, 0.0

    mp_pose = mp.solutions.pose
    img_rgb = np.array(pil_img.convert("RGB"))

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(img_rgb)

    if not results.pose_landmarks:
        return None, 0.0

    nose_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
    ankles = [
        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y,
        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
    ]
    lowest_ankle_y = max(ankles)

    # y 座標は画像高さで正規化されており、下に行くほど値が大きい。
    is_upright = nose_y < lowest_ankle_y
    confidence = float(abs(lowest_ankle_y - nose_y))

    return (0 if is_upright else 180), confidence


def determine_output_path(inp: Path) -> Path:
    """Return output path with ``_rot`` suffix in the same directory."""

    return inp.with_name(f"{inp.stem}_rot{inp.suffix}")


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


def main(inp):
    inp = Path(inp)
    out = determine_output_path(inp)

    images = convert_from_path(str(inp), dpi=DPI)
    pdf = pikepdf.open(str(inp))

    low = []
    changed = 0
    high_conf_rots = []

    for i, (img, page) in enumerate(zip(images, pdf.pages), start=1):
        # Step1: 横長ページは先に90/270に丸めて縦向きにする
        initial_rot, initial_conf = detect_rotation_osd(img)
        is_landscape = img.width > img.height
        first_allowed = [90, 270] if is_landscape else [0, 180]
        snapped_first = snap_rotation_to_allowed(initial_rot, first_allowed)

        portrait_img = img.rotate(-snapped_first, expand=True) if is_landscape else img

        # Step2: 縦向き前提で上下(0/180)を推定する
        upright_rot, upright_conf = detect_rotation_osd(portrait_img)
        upright_snap = snap_rotation_to_allowed(upright_rot, [0, 180])

        used_pose = False
        if upright_conf < CONF_THRESHOLD and not page_contains_text(portrait_img):
            pose_rot, pose_conf = estimate_updown_with_pose(portrait_img)
            if pose_rot is not None and pose_conf >= upright_conf:
                upright_snap = pose_rot
                upright_conf = max(pose_conf, CONF_THRESHOLD)
                used_pose = True

        total_rotation = (snapped_first + upright_snap) % 360 if is_landscape else upright_snap

        # 信頼度が低いページは、これまでの高信頼ページの多数決で補完する
        used_fallback = False
        if upright_conf < CONF_THRESHOLD and high_conf_rots:
            total_rotation = Counter(high_conf_rots).most_common(1)[0][0]
            used_fallback = True
        else:
            if upright_conf >= CONF_THRESHOLD:
                high_conf_rots.append(total_rotation)

        if total_rotation % 360 != 0:
            page.Rotate = total_rotation
            changed += 1

        if upright_conf < CONF_THRESHOLD:
            low.append(
                (
                    i,
                    total_rotation,
                    upright_conf,
                    used_fallback,
                    snapped_first,
                    used_pose,
                )
            )

    out = save_pdf(pdf, out)

    print(f"Saved: {out}  changed_pages={changed}")
    if low:
        print("Low-confidence pages (please verify):")
        for p, r, c, used_fallback, snapped_first, used_pose in low:
            suffix = " (fallback used)" if used_fallback else ""
            pose_suffix = " (pose used)" if used_pose else ""
            print(
                f"  page {p}: rot={r}, conf={c}{suffix}{pose_suffix}, initial_landscape_rot={snapped_first}"
            )


if __name__ == "__main__":
    import sys

    # コマンドライン引数があれば従来通り使う。なければファイルダイアログで選択。
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    else:
        root = tk.Tk()
        root.withdraw()
        input_file = filedialog.askopenfilename(
            title="回転を補正するPDFを選択してください",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if not input_file:
            print("入力ファイルが選択されなかったため終了します。")
            raise SystemExit(1)

    main(input_file)
