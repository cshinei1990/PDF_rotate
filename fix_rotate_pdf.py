from pathlib import Path
from collections import Counter

import numpy as np

import pikepdf
from pdf2image import convert_from_path
import pytesseract
import tkinter as tk
from tkinter import filedialog

DEFAULT_CONF_THRESHOLD = 5.0   # 漫画向けに低め。まずはこのくらいから
DEFAULT_DPI = 200              # 低すぎるとOSD精度が落ちやすい

# 実行時に入力で上書きされる値
CONF_THRESHOLD = DEFAULT_CONF_THRESHOLD
DPI = DEFAULT_DPI


def detect_rotation_osd(pil_img, lang="jpn+eng"):
    try:
        # Use dict output to avoid regex parsing and convert to RGB to ensure DPI metadata
        # script detection (OSD) works better with appropriate languages, though mostly script-independent
        osd = pytesseract.image_to_osd(
            pil_img.convert("RGB"),
            lang=lang,
            output_type=pytesseract.Output.DICT
        )
        rot = int(osd.get("rotate", 0))
        conf = float(osd.get("orientation_confidence", 0.0))
    except pytesseract.TesseractError as exc:
        # When Tesseract fails due to few characters or missing resolution, treat as unknown
        print(f"Tesseract OSD failed ({exc}); using default rotation=0, conf=0")
        rot, conf = 0, 0.0

    return rot, conf


def detect_pose_up_down(pil_img):
    """Use a pose-estimation model to infer whether the image is upside-down.

    Returns a tuple of (rotation, confidence). If the library is unavailable or
    no pose is detected, ``(0, 0.0)`` is returned.
    """

    try:
        import mediapipe as mp  # type: ignore
    except ImportError:
        print("mediapipe が見つからなかったため、姿勢による上下判定はスキップします。")
        return 0, 0.0

    pose = mp.solutions.pose.Pose(static_image_mode=True, enable_segmentation=False)
    rgb = pil_img.convert("RGB")
    height = rgb.size[1]
    results = pose.process(np.array(rgb))

    if not results.pose_landmarks:
        return 0, 0.0

    landmarks = results.pose_landmarks.landmark
    key_indices = [mp.solutions.pose.PoseLandmark.NOSE, mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_HIP]
    if any(landmarks[idx].visibility < 0.3 for idx in key_indices):
        return 0, 0.0

    nose_y = landmarks[mp.solutions.pose.PoseLandmark.NOSE].y * height
    hip_y = (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP].y + landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y) / 2 * height

    if nose_y > hip_y:
        return 180, 10.0
    return 0, 10.0


def has_text_content(pil_img) -> bool:
    """Return True if Tesseract finds any text-like content on the image."""

    try:
        # Support both horizontal (jpn/eng) and vertical (jpn_vert) text detection
        data = pytesseract.image_to_data(
            pil_img.convert("RGB"),
            lang="jpn+jpn_vert+eng",
            output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractError:
        return False

    texts = data.get("text", [])
    confs = data.get("conf", [])
    for text, conf in zip(texts, confs):
        if text.strip() and conf not in (-1, "-1"):
            return True
    return False


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


def prompt_numeric_value(name: str, default, caster):
    message = (
        f"{name} を入力してください（参考値: {default}）。未入力の場合は {default} を使用します: "
    )
    raw = input(message).strip()
    if not raw:
        return default

    try:
        return caster(raw)
    except (TypeError, ValueError):
        print(f"{name} の入力を解釈できませんでした。参考値 {default} を使用します。")
        return default


def process_file(inp: Path) -> None:
    out = determine_output_path(inp)

    images = convert_from_path(str(inp), dpi=DPI)
    pdf = pikepdf.open(str(inp))

    low = []
    changed = 0
    high_conf_updown_rots = []

    for i, (img, page) in enumerate(zip(images, pdf.pages), start=1):
        current_rotation = int(page.obj.get("/Rotate", 0))

        # --- Step 1: rotate landscape pages into portrait (90 or 270) ---
        is_portrait = img.height >= img.width
        primary_rot = 0
        primary_conf = 10.0 if is_portrait else 0.0

        if not is_portrait:
            rot_landscape, conf_landscape = detect_rotation_osd(img)
            primary_rot = snap_rotation_to_allowed(rot_landscape, [90, 270])
            primary_conf = conf_landscape

        portrait_img = img.rotate(-primary_rot, expand=True) if primary_rot else img

        # --- Step 2: decide upright vs upside-down (0 or 180) ---
        used_fallback = False
        rot, conf = 0, 0.0

        if has_text_content(portrait_img):
            rot, conf = detect_rotation_osd(portrait_img)

        if conf < CONF_THRESHOLD:
            # OSDの信頼度が低い場合（またはテキストがない場合）、姿勢推定を試す
            # ただし、テキストが検出されている場合は OSD を優先したいので、
            # OSD の信頼度が極端に低い (例: < 0.5) 場合のみ姿勢判定を採用する
            run_pose = True
            if has_text_content(portrait_img) and conf >= 0.5:
                run_pose = False

            if run_pose:
                p_rot, p_conf = detect_pose_up_down(portrait_img)
                if p_conf > conf:
                    rot = p_rot
                    conf = p_conf

        allowed_updown = [0, 180]
        if conf < CONF_THRESHOLD and high_conf_updown_rots:
            applied_updown = Counter(high_conf_updown_rots).most_common(1)[0][0]
            used_fallback = True
        else:
            applied_updown = snap_rotation_to_allowed(rot, allowed_updown)

        if conf >= CONF_THRESHOLD:
            high_conf_updown_rots.append(applied_updown)

        # 縦長ページかつ回転操作(180度)が加わる場合、ダブルチェックを行う
        # 元画像(0度)と回転後画像(180度)でOSDを行い、どちらが「0度(正立)」と判定されるか、かつその信頼度を比較する
        if is_portrait and applied_updown != 0:
            # 1. 元画像の正立度合い
            # すでに計算済みの rot, conf を利用する
            # rotが0なら「現在は正立している」という判定なので、その信頼度をスコアにする
            # rotが180なら「現在は倒立している」という判定なので、正立スコアは0とみなす(あるいは低い値)
            score_original = conf if rot == 0 else 0.0

            # テキスト未検出などでOSDをまだやっていない場合は実行する
            if not has_text_content(portrait_img):
                 # テキストが無いと判定されていても、強制的にOSDを試みる（または何もしない？）
                 # ここでは「OSD判定を行い」という指示なので試行する
                 check_rot, check_conf = detect_rotation_osd(portrait_img)
                 score_original = check_conf if check_rot == 0 else 0.0

            # 2. 回転後画像の正立度合い
            rotated_img = portrait_img.rotate(applied_updown, expand=True)
            r_rot, r_conf = detect_rotation_osd(rotated_img)
            score_rotated = r_conf if r_rot == 0 else 0.0

            # 元の方が「正立している」信頼度が高いなら、回転を取り消す
            if score_original > score_rotated:
                print(f"  [DoubleCheck] page {i}: Reverting 180 rotation. Score Orig({score_original}) > Rot({score_rotated})")
                applied_updown = 0
                total_rotation = (primary_rot + applied_updown) % 360 # Update if needed, though primary is 0 here
            else:
                 # 回転後の方が良い、あるいはどっちもダメなら当初の判定(Poseなど)を優先
                 pass

        total_rotation = (primary_rot + applied_updown) % 360
        new_rotation = (current_rotation + total_rotation) % 360

        if new_rotation != current_rotation:
            page.Rotate = new_rotation
            changed += 1

        if conf < CONF_THRESHOLD or primary_conf < CONF_THRESHOLD:
            low.append((i, primary_rot, applied_updown, conf, used_fallback, total_rotation))

    out = save_pdf(pdf, out)

    print(f"Saved: {out}  changed_pages={changed}")
    if low:
        print("Low-confidence pages (please verify):")
        for p, primary, updown, c, used_fallback, total in low:
            suffix = " (fallback used)" if used_fallback else ""
            print(f"  page {p}: total_rot={total} (portrait_fix={primary}, updown={updown}), conf={c}{suffix}")


if __name__ == "__main__":
    import sys

    CONF_THRESHOLD = prompt_numeric_value("CONF_THRESHOLD", DEFAULT_CONF_THRESHOLD, float)
    DPI = prompt_numeric_value("DPI", DEFAULT_DPI, int)

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
