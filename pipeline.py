"""
Receipt OCR Pipeline using PaddleOCR
Carbon Crunch Internship Assignment
"""

import cv2
import numpy as np
import json
import re
import os
from pathlib import Path
from paddleocr import PaddleOCR
from datetime import datetime
from typing import Optional


# ── Init OCR (once, reused across calls) ────────────────────────────────────
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')


# ════════════════════════════════════════════════════════════════════════════
# 1. IMAGE PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════


def _crop_to_content(img: np.ndarray, padding: int = 20) -> np.ndarray:
    """Remove large empty borders by finding the bounding box of non-white content."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold: anything not near-white is content
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img  # nothing to crop, return original
    x, y, w, h = cv2.boundingRect(coords)
    # Add padding, clamp to image bounds
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img.shape[1], x + w + padding)
    y2 = min(img.shape[0], y + h + padding)
    return img[y1:y2, x1:x2]


def _fix_orientation(img: np.ndarray) -> np.ndarray:
    """Detect upside-down receipts using a quick full-image OCR pass."""
    import tempfile

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)
    cv2.imwrite(tmp_path, img)

    try:
        r = ocr_engine.ocr(tmp_path)
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
def preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    # ── Resize if too large (PaddleOCR crashes on very large images) ─────────
    MAX_DIM = 1920
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    # ── Crop whitespace (remove large empty borders around receipt) ───────────
    img = _crop_to_content(img)

    img = _fix_orientation(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10,
                                         templateWindowSize=7,
                                         searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(denoised)
    deskewed = _deskew(contrasted)

    return cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)

 



def _deskew(gray: np.ndarray) -> np.ndarray:
    """Correct skew using Hough lines."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return gray

    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:   # negligible skew
        return gray

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# ════════════════════════════════════════════════════════════════════════════
# 2. OCR
# ════════════════════════════════════════════════════════════════════════════
def sort_lines_by_position(lines: list[dict]) -> list[dict]:
    """
    Sort OCR lines by vertical position (top to bottom),
    then merge lines on the same row into one text entry.
    This fixes multi-column receipts where qty/item/price
    are detected as separate boxes on the same horizontal row.
    """
    if not lines:
        return lines

    # Get top-y of each bounding box
    def get_top_y(line):
        bbox = line["bbox"]
        return min(pt[1] for pt in bbox)

    def get_left_x(line):
        bbox = line["bbox"]
        return min(pt[0] for pt in bbox)

    # Sort top-to-bottom first
    sorted_lines = sorted(lines, key=get_top_y)

    # Group lines that are on the same row (within Y_TOLERANCE pixels)
    Y_TOLERANCE = 10
    rows = []
    current_row = [sorted_lines[0]]

    for line in sorted_lines[1:]:
        current_top = get_top_y(line)
        row_top = get_top_y(current_row[0])
        if abs(current_top - row_top) <= Y_TOLERANCE:
            current_row.append(line)
        else:
            rows.append(current_row)
            current_row = [line]
    rows.append(current_row)

    # Within each row, sort left-to-right and merge into one line
    # Within each row, sort left-to-right and dynamically inject spaces
    merged = []
    for row in rows:
        row_sorted = sorted(row, key=get_left_x)
        
        merged_text = ""
        prev_right_x = None
        
        for l in row_sorted:
            bbox = l["bbox"]
            left_x = min(pt[0] for pt in bbox)
            right_x = max(pt[0] for pt in bbox)
            
            if prev_right_x is not None:
                # Calculate the physical pixel gap between the previous word and this word
                pixel_gap = left_x - prev_right_x
                
                # Convert pixels to terminal spaces (Assume ~12 pixels per character)
                # Ensure at least 1 space exists if they are on the same line
                space_count = max(1, int(pixel_gap / 12))
                merged_text += " " * space_count
                
            merged_text += l["text"]
            prev_right_x = right_x

        avg_conf = round(sum(l["confidence"] for l in row_sorted) / len(row_sorted), 4)
        
        merged.append({
            "text": merged_text,
            "confidence": avg_conf,
            "bbox": row_sorted[0]["bbox"] # Keep the leftmost anchor
        })

    return merged

def run_ocr(image_input) -> list[dict]:
    result = ocr_engine.ocr(image_input)
    lines = []

    if not result:
        return lines

    for page in result:
        if isinstance(page, dict):
            texts  = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            polys  = page.get("rec_polys", [])
            for text, score, bbox in zip(texts, scores, polys):
                if text.strip():
                    lines.append({
                        "text":       text.strip(),
                        "confidence": round(float(score), 4),
                        "bbox":       bbox.tolist()
                    })

    # ← Sort spatially and merge same-row fragments
    lines = sort_lines_by_position(lines)
    return lines

    
# ════════════════════════════════════════════════════════════════════════════
# 3. FUZZY MATCHING HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _char_similarity(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    if not a or not b:
        return 0.0
    matches = sum(ca == cb for ca, cb in zip(a, b))
    return matches / max(len(a), len(b))

def _fuzzy_contains(text: str, keywords: set, threshold: float = 0.65) -> bool:
    """Lower threshold (0.65) to catch TOIAI→TOTAL, SUBIOIA→SUBTOTAL."""
    words = re.findall(r"[a-z]+", text.lower())
    for kw in keywords:
        kw_words = kw.split()
        if len(kw_words) == 1:
            for w in words:
                if _char_similarity(w, kw) >= threshold:
                    return True
        else:
            if _char_similarity(text.lower()[:len(kw)], kw) >= threshold:
                return True
    return False

KNOWN_STORES = [
    "walmart", "walgreens", "target", "costco", "kroger",
    "whole foods", "trader joe", "cvs", "aldi", "safeway",
    "publix", "meijer", "heb", "lidl", "dollar", "sams club",
]

def _is_likely_store_name(text: str) -> tuple[bool, float]:
    t = text.lower().strip()
    for store in KNOWN_STORES:
        if _char_similarity(t, store) >= 0.75:
            return True, 0.15
    if re.fullmatch(r"[A-Za-z\s\.\*\-&']{3,25}", text):
        return True, 0.05
    return False, 0.0


# ════════════════════════════════════════════════════════════════════════════
# 4. KEY INFORMATION EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

DATE_PATTERNS = [
    r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
    r"\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b",
    r"\b([A-Za-z]+ \d{1,2},?\s*\d{4})\b",
]
# PRICE_PATTERN = r"\$?\s*(\d{1,4}[.,]\d{2})"   # removed \b — catches "50.00 O"
TOTAL_KEYWORDS = {"total", "amount due", "grand total", "balance due",
                  "total amount", "net amount", "amount payable"}
# ITEM_PRICE_RE  = re.compile(
#     r"^(.+?)\s+\$?\s*(\d{1,4}[.,]\d{2})"   # relaxed — no strict end anchor
# )
SKIP_KEYWORDS  = {"subtotal", "tax", "discount", "change", "cash", 
                  "tend", "you saved", "time", "date"}

# NON_ITEM_PATTERNS = [
#     r"^\d{6,}$",
#     r"^[A-Z0-9#\*\/]{8,}$",
#     r"@", r"survey|feedback|thank|www\.|\.com",
#     r"^\s*LD\s*#",   
#     r"\b\d{2}:\d{2}\b", # Block digital timestamps like 23:37 or 10:46
# ]



# # ── Constants to add ─────────────────────────────────────────────────────────
# HEADER_SKIP_PATTERNS = re.compile(
#     r"see back|chance to win|\$1000|id\s*#|save money|live better"
#     r"|always low|open 24|supercenter|manager|always\."
#     r"|scan with|store receipts|low prices|thank you"
#     r"|pay from primary|eft debit|us debit|visa|ref #"
#     r"|terminal|network id|appr|aid |items sold"
#     r"|sharon rd|save money",
#     re.I
# )


# ── Fix 1: extract_store_name ─────────────────────────────────────────────────
def extract_store_name(lines: list[dict]) -> dict:
    if not lines:
        return {"value": None, "confidence": 0.0, "flag": "LOW_CONFIDENCE"}

    # Filter out promotional/footer lines
    valid_lines = [
        l for l in lines
        if len(l["text"].strip()) > 4
        and not HEADER_SKIP_PATTERNS.search(l["text"])
    ]

    if not valid_lines:
        return {"value": None, "confidence": 0.0, "flag": "LOW_CONFIDENCE"}

    # Pass 1: known store names anywhere in first 10 valid lines
    for line in valid_lines[:10]:
        t = line["text"].strip()
        is_store, bonus = _is_likely_store_name(t)
        if is_store:
            conf = min(1.0, line["confidence"] + bonus)
            result = {"value": t, "confidence": round(conf, 3)}
            if conf < 0.7:
                result["flag"] = "LOW_CONFIDENCE"
            return result

    # Pass 2: first clean alphabetic line
    for line in valid_lines[:6]:
        t = line["text"].strip()
        alpha_ratio = len(re.findall(r"[A-Za-z]", t)) / max(len(t), 1)
        if alpha_ratio > 0.6 and not re.search(r"^\d", t):
            conf = line["confidence"]
            result = {"value": t, "confidence": round(conf, 3)}
            if conf < 0.7:
                result["flag"] = "LOW_CONFIDENCE"
            return result

    return {"value": None, "confidence": 0.0, "flag": "LOW_CONFIDENCE"}


# ── Fix 2: extract_date ───────────────────────────────────────────────────────
# Add this validation function
# Fix 2 — _is_valid_date: allow years 00-99 (2000s receipts)
def _is_valid_date(value: str) -> bool:
    parts = re.split(r"[\/\-\.]", value)
    if len(parts) != 3:
        return False
    try:
        a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return False

    # Normalise 2-digit year
    year = c if len(str(c)) <= 4 and len(parts[2]) >= 2 else a
    if len(parts[2]) == 2:
        year += 2000  # treat "10" as 2010, "99" as 2099

    if not (2000 <= year <= 2099):
        return False

    # MM/DD/YY or DD/MM/YY — just check ranges are plausible
    if not (1 <= a <= 31 and 1 <= b <= 12):
        if not (1 <= b <= 31 and 1 <= a <= 12):
            return False

    return True

def extract_date(lines: list[dict]) -> dict:
    # Pass 1: lines with explicit "Date" label — most reliable
    for line in lines:
        t = line["text"]
        if re.search(r"\bdate\b", t, re.I):
            for pat in DATE_PATTERNS:
                m = re.search(pat, t)
                if m and _is_valid_date(m.group(1)):
                    conf = min(1.0, line["confidence"] * 0.85 + 0.15)
                    return {"value": m.group(1), "confidence": round(conf, 3)}

    # Pass 2: lines with timestamp (date + time together) — very reliable
    for line in lines:
        t = line["text"]
        if re.search(r"\d{2}:\d{2}", t):  # has a time component
            for pat in DATE_PATTERNS:
                m = re.search(pat, t)
                if m and _is_valid_date(m.group(1)):
                    conf = min(1.0, line["confidence"] * 0.85 + 0.15)
                    return {"value": m.group(1), "confidence": round(conf, 3)}

    # Pass 3: any line with a valid date
    for line in lines:
        for pat in DATE_PATTERNS:
            m = re.search(pat, line["text"])
            if m and _is_valid_date(m.group(1)):
                conf = min(1.0, line["confidence"] * 0.85 + 0.15)
                return {"value": m.group(1), "confidence": round(conf, 3)}

    return {"value": None, "confidence": 0.0, "flag": "LOW_CONFIDENCE"}


# ── Fix 3: extract_items — handle receipts with NO column header ──────────────
# 1. Update the Constants (Accept 2 or 3 decimals, make space optional)
PRICE_PATTERN = r"\$?\s*(\d{1,4}[.,]\d{2,3})"
# Fix 1 — ITEM_PRICE_RE: require space before price
ITEM_PRICE_RE = re.compile(
    r"^(.+?)\s+\$?\s*(\d{1,4}[.,]\d{2,3})"  # \s+ required — no optional space
)


# # 2. Update the Extraction Function
# def classify_layout(lines: list[dict], start_idx: int) -> str:
#     """Determine if receipt is Linear (Name first) or Tabular (Price first, Name below)."""
#     for i in range(start_idx, min(start_idx + 10, len(lines))):
#         t = lines[i]["text"]
#         prices = re.findall(PRICE_PATTERN, t)
        
#         if prices:
#             # We found the first line with prices. 
#             # Check the line directly below it.
#             if i + 1 < len(lines):
#                 next_text = lines[i+1]["text"]
#                 alpha_ratio = len(re.findall(r"[A-Za-z]", next_text)) / max(len(next_text), 1)
#                 next_prices = re.findall(PRICE_PATTERN, next_text)
                
#                 # If the line BELOW the prices is heavy text and has no prices, it's Tabular.
#                 if alpha_ratio > 0.5 and not next_prices:
#                     return "TABULAR"
            
#             return "LINEAR"
            
#     return "LINEAR" # Default fallback


# # ── Updated constants ─────────────────────────────────────────────────────────

HEADER_SKIP_PATTERNS = re.compile(
    r"see back|chance to win|\$1000|id\s*#|save money|live better"
    r"|always low|open 24|supercenter|manager|always\."
    r"|scan with|store receipts|low prices|thank you"
    r"|pay from primary|eft debit|us debit|visa|ref #"
    r"|terminal|network id|appr|aid |items sold"
    r"|sharon rd|save money|www\.|\.com|http"
    r"|returns|purchases|amazon|earn \d|learn more"
    r"|open \d|store #|cashier|salesperson|member"
    r"|entry time|exit time|parking",   # ← parking receipt labels
    re.I
)
# Fix 2 — add to NON_ITEM_LINE_RE: weight annotation lines and @ lines
NON_ITEM_LINE_RE = re.compile(
    r"^\s*\d+\s*@\s*[\d.]"             # "2 @ 0.49"
    r"|^\s*\d+@[\d.]"                  # "2@0.49" — no space variant ← NEW
    r"|^\s*@"                           # any line starting with @
    r"|\d+\s*(lb|1b|kg|oz)\s*@\s*[\d.]"  # "0.41 lb @ 1 lb/0.49"
    r"|\d+\s*/\s*\d+\.\d+"             # "1 lb /0.49"
    r"|you\s+saved|was\s+\d"
    r"|voided|void"
    r"|\bbal\b|\btax\b.*\bbal\b"
    r"|\bentry\s+time\b|\bexit\s+time\b"
    r"|\brounding\b|\badjustment\b"
    r"|\bgst\s+summary\b|\btax\s+code\b"
    r"|\bnet\s+sales\b|\bsold\s+items\b"
    r"|\bearned\b|\bpoints\b"
    r"|\b\d+ea\b"
    r"|\d+\s*@\s*[\d.]",               # "2@0.69" anywhere in line ← NEW
    re.I
)

# ── Updated extract_total ─────────────────────────────────────────────────────
def extract_total(lines: list[dict]) -> dict:
    """
    Priority: Total Sales Inclusive > Net Total > Grand Total > TOTAL
    Avoids: subtotal, tax, discount, rounding, cash, change lines
    """
    BLOCK_EXACT = {
        "subtotal", "sub total", "total tax", "total gst",
        "total qty", "total sales excl", "total sales excluding",
        "discount", "rounding", "cash", "change", "cash tend",
        "mcard tend", "visa tend", "debit tend",
        "tax 1", "tax 2", "tax 3",
    }

    PREFERRED_KEYWORDS = [
        "total sales inclusive",
        "rounded total",
        "net total",
        "grand total",
        "total amount",
        "amount due",
        "total :",       # Whole Foods format "Total :"
        "total:",
    ]
    FALLBACK_KEYWORD = "total"

    def is_blocked(text_lower):
        for b in BLOCK_EXACT:
            if b in text_lower:
                return True
        return False

    # Pass 1: preferred keywords with price on same line
    for kw in PREFERRED_KEYWORDS:
        for line in lines:
            tl = line["text"].lower()
            if kw in tl and not is_blocked(tl):
                prices = re.findall(PRICE_PATTERN, line["text"])
                if prices:
                    conf = min(1.0, line["confidence"] * 0.85 + 0.15)
                    return {"value": prices[-1].replace(",", "."),
                            "confidence": round(conf, 3)}

    # Pass 2: plain "TOTAL" with price on same line, not blocked
    for line in lines:
        tl = line["text"].lower()
        if re.search(r"\btotal\b", tl) and not is_blocked(tl):
            prices = re.findall(PRICE_PATTERN, line["text"])
            if prices:
                conf = min(1.0, line["confidence"] * 0.85 + 0.15)
                return {"value": prices[-1].replace(",", "."),
                        "confidence": round(conf, 3)}

    # Pass 3: spatial sweep for "TOTAL" label with price on adjacent line
    for line in lines:
        tl = line["text"].lower().strip()
        if re.fullmatch(r"total\s*:?", tl) and not is_blocked(tl):
            a_bbox = line["bbox"]
            anchor_y = (a_bbox[0][1] + a_bbox[3][1]) / 2
            anchor_x = max(a_bbox[1][0], a_bbox[2][0])
            for other in lines:
                if other == line:
                    continue
                ob = other["bbox"]
                oy = (ob[0][1] + ob[3][1]) / 2
                ox = min(ob[0][0], ob[3][0])
                if abs(oy - anchor_y) <= 15 and ox > anchor_x - 20:
                    prices = re.findall(PRICE_PATTERN, other["text"])
                    if prices:
                        conf = min(1.0, other["confidence"] * 0.85 + 0.15)
                        return {"value": prices[-1].replace(",", "."),
                                "confidence": round(conf, 3)}

    return {"value": None, "confidence": 0.0, "flag": "LOW_CONFIDENCE"}


# ── Updated extract_items ─────────────────────────────────────────────────────
# ── Updated extract_items (replaces classify_layout + old extract_items) ──────
def extract_items(lines: list[dict]) -> list[dict]:
    items = []

    HEADER_RE = re.compile(
        r"\b(item|desc|description)\b.{0,50}\b(price|amt|amount)\b"
        r"|\bqty\b.{0,40}\b(price|amount|amt|s/price)\b"
        r"|\bs[/\\]price\b|\bcode\s*/\s*desc\b",
        re.I
    )

    # Payment/footer keywords — zone exit triggers
    PAYMENT_ZONE_RE = re.compile(
        r"^\s*(cash|change|tend|mcard|visa|debit|account\s*#"
        r"|approval|trans\s*id|validation|payment\s*service"
        r"|ref\s*#|network|terminal|aid\s|tc#|items\s*sold)\b",
        re.I
    )

    has_header = any(HEADER_RE.search(l["text"]) for l in lines)
    in_item_zone = False

    if not has_header:
        for i, line in enumerate(lines):
            if re.search(r"\bST#\b|\bOP#\b", line["text"]):
                in_item_zone = True
                lines = lines[i+1:]
                break
        if not in_item_zone:
            in_item_zone = True

    for i, line in enumerate(lines):
        t = line["text"]
        lower_t = t.lower()

        # Zone entry
        if has_header and HEADER_RE.search(lower_t):
            in_item_zone = True
            continue

        # Hard zone exit — tax rate line or final total
        if re.search(r"\btax\s*\d|\btax\s+\d+\.\d+\s*%", lower_t):
            break
        if re.search(
            r"\btotal\s+(sales|amount|inclusive|exclusive|payable|tend|tax)\b"
            r"|\bnet\s+total\b|\bgrand\s+total\b|\brounded\s+total\b",
            lower_t, re.I
        ):
            break

        # Soft skip — subtotal, don't exit zone
        if re.search(r"\bsubtotal\b|\bsub\s+total\b", lower_t):
            continue

        # Payment section — exit zone immediately
        if PAYMENT_ZONE_RE.search(t):
            break

        if not in_item_zone:
            continue

        # Skip clearly non-item lines
        if HEADER_SKIP_PATTERNS.search(t):
            continue
        if NON_ITEM_LINE_RE.search(t):
            continue
        if re.search(r"\b(ST#|OP#|TE#|TR#)\b", t):
            continue
        if re.match(r"^\s*\d+\s*(ea|pc|lb|kg|x)\s", lower_t):
            continue
        # Add this skip check in extract_items after the existing re.match for ea/pc/lb:
        if re.match(r"^\s*[\d.]+\s*(1b|lb|kg|oz)\s*[@\d]", lower_t):
            continue
        if re.search(r"voided|void\s+entry|discount\s+given", lower_t):
            continue

        prices = re.findall(PRICE_PATTERN, t)

        # Split-line: current line has name only, next line has price only
        if not prices and i + 1 < len(lines):
            next_t = lines[i+1]["text"].strip()
            next_prices = re.findall(PRICE_PATTERN, next_t)
            has_alpha = len(re.findall(r"[A-Za-z]", t)) > 2
            next_mostly_numeric = len(re.findall(r"[A-Za-z]", next_t)) == 0 or \
                      re.fullmatch(r"[\d.,\s]+[NFXBE]?", next_t.strip())
            if has_alpha and next_prices and next_mostly_numeric \
               and not NON_ITEM_LINE_RE.search(t) \
               and not HEADER_SKIP_PATTERNS.search(t)\
               and not re.search(r"\d+@", next_t):
                name = re.sub(r"\s+\d{6,}\s*[A-Z]?\s*$", "", t).strip()
                name = re.sub(r"[©®™\*]", "", name).strip()
                if len(name) > 2:
                    price = next_prices[-1].replace(",", ".")
                    conf = min(1.0, line["confidence"] * 0.9 + 0.05)
                    entry = {"name": name, "price": price, "confidence": round(conf, 3)}
                    if conf < 0.7:
                        entry["flag"] = "LOW_CONFIDENCE"
                    items.append(entry)
            continue

        if not prices:
            continue

        m = ITEM_PRICE_RE.match(t)
        name = m.group(1).strip() if m else ""

        # Fix 3 — barcode stripping: strip ALL trailing barcodes not just 6+ digits
        # In extract_items, replace the name cleaning block:

        name = re.sub(r"\s+\d{5,}\s*[A-Z]?\s*$", "", name).strip()  # trailing barcode
        name = re.sub(r"\s+\d{5,}\s*[A-Z]?\s*", " ", name).strip()  # inline barcode
        name = re.sub(r"[©®™\*]", "", name).strip()
        name = re.sub(r"^\*?(VC|WT|wt)\s+", "", name).strip()
        name = re.sub(r"^\*+\s*", "", name).strip()
        name = re.sub(r"\s+[NFXTBE]S?$", "", name).strip()

        alpha_ratio = len(re.findall(r"[A-Za-z]", name)) / max(len(name), 1) if name else 0

        if alpha_ratio < 0.3 and i > 0:
            prev_t = lines[i-1]["text"].strip()
            prev_prices = re.findall(PRICE_PATTERN, prev_t)
            prev_alpha = len(re.findall(r"[A-Za-z]", prev_t)) / max(len(prev_t), 1)
            if prev_alpha > 0.4 and not prev_prices and not NON_ITEM_LINE_RE.search(prev_t):
                name = prev_t

        # Replace the lookahead block in extract_items with this:

        if i + 1 < len(lines):
            next_t = lines[i+1]["text"].strip()
            next_prices = re.findall(PRICE_PATTERN, next_t)
            next_alpha = len(re.findall(r"[A-Za-z]", next_t)) / max(len(next_t), 1)

            # Only replace name with next line if current name is genuinely weak
            current_name_weak = (
                len(name) < 4 or
                len(re.findall(r"[A-Za-z]", name)) / max(len(name), 1) < 0.4
            )

            if current_name_weak \
            and not next_prices \
            and next_alpha > 0.6 \
            and len(next_t) > 3 \
            and not NON_ITEM_LINE_RE.search(next_t) \
            and not HEADER_SKIP_PATTERNS.search(next_t):
                name = next_t

        name = re.sub(r"^[\W_]+", "", name).strip()

        if len(name) < 2:
            continue
        if re.search(
            r"\b(qty|item|price|amount|cashier|salesperson|manager"
            r"|st#|subtotal|tax|total|discount|rounding|entry|exit"
            r"|tare|item\s*=|doc\s*no|invoice|ref|cash|change|tend)\b",
            name, re.I
        ):
            continue
        if len(re.findall(r"\d", name)) / max(len(name), 1) > 0.6:
            continue

        # Skip negative prices (discounts)
        price_pos = t.rfind(prices[-1])
        after_price = t[price_pos + len(prices[-1]):].strip()
        if after_price.startswith("-"):
            continue

        price = prices[-1].replace(",", ".")
        conf = min(1.0, line["confidence"] * 0.9 + 0.05)
        entry = {"name": name, "price": price, "confidence": round(conf, 3)}
        if conf < 0.7:
            entry["flag"] = "LOW_CONFIDENCE"
        items.append(entry)

    # Fix 4 — deduplicate items (Trader Joe's lookahead creates duplicates)
# Add this at the END of extract_items before return:

# Deduplicate: remove items with same name+price added consecutively
    seen = set()
    deduped = []
    for entry in items:
        key = (entry["name"].lower().strip(), entry["price"])
        if key not in seen:
            seen.add(key)
            deduped.append(entry)
    items = deduped

    return items
def extract_fields(lines: list[dict]) -> dict:
    return {
        "store_name":   extract_store_name(lines),
        "date":         extract_date(lines),
        "items":        extract_items(lines),
        "total_amount": extract_total(lines),
    }
# ════════════════════════════════════════════════════════════════════════════
# 5. FINANCIAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def generate_summary(results: list[dict]) -> dict:
    """Aggregate across all receipts."""
    total_spend      = 0.0
    transactions     = 0
    spend_per_store  = {}
    failed           = []

    for r in results:
        if "error" in r:
            failed.append(r["source"])
            continue

        transactions += 1
        store = (r["store_name"].get("value") or "Unknown")

        amt_raw = r["total_amount"].get("value")
        try:
            amt = float(amt_raw) if amt_raw else 0.0
        except ValueError:
            amt = 0.0

        total_spend += amt
        spend_per_store[store] = round(
            spend_per_store.get(store, 0.0) + amt, 2
        )

    return {
        "num_transactions": transactions,
        "total_spend":      round(total_spend, 2),
        "spend_per_store":  spend_per_store,
        "failed_images":    failed,
    }


# ════════════════════════════════════════════════════════════════════════════
# 6. BATCH RUNNER
# ════════════════════════════════════════════════════════════════════════════

def run_batch(image_dir: str, output_dir: str = "outputs") -> None:
    import tempfile
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = [
        str(p) for p in Path(image_dir).iterdir()
        if p.suffix.lower() in supported
    ]

    if not image_paths:
        print("No images found.")
        return

    all_results = []
    for path in sorted(image_paths):
        print(f"\n[+] Processing: {path}")
        try:
            img = preprocess_image(path)

            # ← Must write to disk — PaddleOCR needs a file path, not numpy array
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(tmp_fd)

            cv2.imwrite(tmp_path, img)
            lines = run_ocr(tmp_path)

            try:
                os.remove(tmp_path)
            except OSError:
                pass

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            all_results.append({"error": str(e), "source": os.path.basename(path)})
            continue

        fields = extract_fields(lines)
        avg_conf = round(np.mean([l["confidence"] for l in lines]), 3) if lines else 0.0

        result = {
            "source":             os.path.basename(path),
            "avg_ocr_confidence": avg_conf,
            "store_name":         fields["store_name"],
            "date":               fields["date"],
            "items":              fields["items"],
            "total_amount":       fields["total_amount"],
        }
        all_results.append(result)

        out_path = Path(output_dir) / (Path(path).stem + ".json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"    → Saved: {out_path}")

    summary = generate_summary(all_results)
    summary_path = Path(output_dir) / "financial_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Processed : {len(all_results)} receipts")
    print(f"  Total spend: ${summary['total_spend']}")
    print(f"{'='*50}\n")

# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    img_dir = sys.argv[1] if len(sys.argv) > 1 else "images"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    run_batch(img_dir, out_dir)