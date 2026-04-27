import os 
import cv2
import numpy as np
from src.ocr import get_engine

MAX_DIM = 1920


def _fix_orientation(img: np.ndarray) -> np.ndarray:
    """Detect upside-down receipts using a quick full-image OCR pass."""
    import tempfile

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)
    cv2.imwrite(tmp_path, img)

    try:
        r = get_engine().ocr(tmp_path)
        all_texts = []
        if r:
            for page in r:
                if isinstance(page, dict):
                    all_texts = page.get("rec_texts", [])
                    break
    except:
        all_texts = []
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if not all_texts:
        return img

    # Check first 5 vs last 5 lines for orientation signals
    first_chunk = " ".join(all_texts[:5]).lower()
    last_chunk  = " ".join(all_texts[-5:]).lower()

    FOOTER = ["low prices", "thank you", "items sold", "tc#",
              "change due", "beg bal", "end bal", "scan with"]
    HEADER = ["walmart", "wal-mart", "wal*mart", "whole foods", "target",
              "walgreens", "kroger", "save money", "always low"]

    footer_at_start = any(k in first_chunk for k in FOOTER)
    header_at_end   = any(k in last_chunk  for k in HEADER)
    header_at_start = any(k in first_chunk for k in HEADER)

    if footer_at_start or header_at_end:
        print("  [INFO] Upside-down receipt detected — rotating 180°")
        return cv2.rotate(img, cv2.ROTATE_180)

    return img


def preprocess(image_path: str) -> np.ndarray:
    img = _load(image_path)
    img = _resize(img)
    img = _crop_to_content(img)
    img = _fix_orientation(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _denoise(gray)
    gray = _clahe(gray)
    gray = _deskew(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def _load(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    return img


def _resize(img: np.ndarray) -> np.ndarray:
    """Downscale if either dimension exceeds MAX_DIM."""
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def _crop_to_content(img: np.ndarray, padding: int = 20) -> np.ndarray:
    """Remove large empty white borders around the receipt."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img.shape[1], x + w + padding)
    y2 = min(img.shape[0], y + h + padding)
    return img[y1:y2, x1:x2]


def _denoise(gray: np.ndarray) -> np.ndarray:
    """Non-local means denoising — preserves text edges better than Gaussian alone."""
    return cv2.fastNlMeansDenoising(gray, h=10,
                                     templateWindowSize=7,
                                     searchWindowSize=21)


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _deskew(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return gray

    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if abs(angle) < 45:   # discard vertical/diagonal lines from borders
            angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return gray

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
