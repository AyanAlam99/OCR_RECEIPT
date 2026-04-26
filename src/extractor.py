"""
src/extractor.py
────────────────
Phase 3 — Spatial Mapping & Heuristic Field Extraction

Responsibilities:
  - Detect receipt layout (LINEAR vs TABULAR)
  - Extract: store_name, date, items, total_amount
  - Each field returned as { value, confidence, ?flag }
"""

import re
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PRICE_RE = re.compile(r"\$?\s*(\d{1,4}[.,]\d{2,3})")

ITEM_PRICE_RE = re.compile(r"^(.+?)\s+\$?\s*(\d{1,4}[.,]\d{2,3})")


DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b"),
    re.compile(r"\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b"),
    re.compile(r"\b([A-Za-z]+ \d{1,2},?\s*\d{4})\b"),
    re.compile(r"\b(\d{1,2}[\/\-\.]\d{2}\d{4})\b"), 
]

KNOWN_STORES = {
    "walmart", "walgreens", "target", "costco", "kroger",
    "whole foods", "trader joe", "cvs", "aldi", "safeway",
    "publix", "meijer", "heb", "lidl", "dollar", "sams club",
}

TOTAL_SKIP = {
    "subtotal", "sub total", "total tax", "total gst",
    "total qty", "total sales excl", "total sales excluding",
    "discount", "rounding", "cash", "change", "cash tend",
    "tax 1", "tax 2", "tax 3","adjustment",
}

TOTAL_PREFERRED = [
    "total sales inclusive", "rounded total", "net total",
    "grand total", "total amount", "amount due", "total :", "total:",
]

ITEM_SKIP_RE = re.compile(
    r"\b(subtotal|sub\s+total|total|gst|tax|discount|rounding"
    r"|cash|change|tend|qty|item|description|price|amount"
    r"|cashier|salesperson|date|time|doc\s*no|invoice)\b",
    re.I,
)

NON_ITEM_LINE_RE = re.compile(
    r"^\s*\d+\s*@\s*[\d.]"
    r"|^\s*@"
    r"|\d+\s*(lb|1b|kg|oz)\s*@"        # existing
    r"|\d+\s*(lb|1b)\s+\d+\s*(lb|1b)"  # ← NEW: "2.51 1b 1 lb" pattern
    r"|you\s+saved|was\s+\d"
    r"|\bvoided?\b"
    r"|\brounding\b|\badjustment\b",
    re.I,
)
HEADER_SKIP_RE= re.compile(
    r"see back|chance to win|\$1000|id\s*#|save money|live better"
    r"|always low|open 24|supercenter|manager|always\."
    r"|scan with|store receipts|low prices|thank you"
    r"|pay from primary|eft debit|us debit|visa|ref #"
    r"|terminal|network id|appr|aid |items sold"
    r"|sharon rd|save money|www\.|\.com|http"
    r"|returns|purchases|amazon|earn \d|learn more"
    r"|open \d|store #|cashier|salesperson|member"
    r"|entry time|exit time|parking"
    r"|www\.|\.com|http|shopping card|card redemption"  # ← NEW
    r"|card tend|debit tend|visa tend|tend\b",
    re.I,
)

PAYMENT_ZONE_RE = re.compile(
        r"\b(cash\s+tend|change\s+due|mcard|visa|debit|account\s*#"
        r"|approval|trans\s*id|validation|payment\s*service"
        r"|ref\s*#|network|terminal|aid\s|tc#|items\s*sold"
        r"|shop.*?card|bg\s*bal|end\s*bal|gift\s*card)\b",
        re.I
    )

# Replaces the old QTY_LINE_RE
QTY_LINE_RE = re.compile(r"^\s*\d+[\.,]?\d*\s*(pc|pcs|kg|nos|x|@)\b", re.I)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def extract_fields(lines: list[dict]) -> dict:
    """Entry point. Returns all four fields with confidence scores."""
    return {
        "store_name":   _extract_store_name(lines),
        "date":         _extract_date(lines),
        "items":        _extract_items(lines),
        "total_amount": _extract_total(lines),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STORE NAME
# ══════════════════════════════════════════════════════════════════════════════

def _extract_store_name(lines: list[dict]) -> dict:
    """
    Score-based extraction — avoids hardcoded word blacklists.
    Scores each of the top 10 lines on structural signals:
      + high OCR confidence
      + mostly alphabetic
      + ALL CAPS (receipts print store names in caps)
      + known store match
      - starts with a digit (address)
      - contains tel/fax/gst/reg patterns
      - label:value format (e.g. "Date : 03/02/2018")
    """
    if not lines:
        return _low_conf_field()

    candidates = lines[:10]
    scored = []

    for i, line in enumerate(candidates):
        t = line["text"].strip()
        if len(t) < 3:
            continue

        score = 0.0
        alpha_ratio = _alpha_ratio(t)

        score += line["confidence"] * 0.3
        score += alpha_ratio * 0.25

        t_clean = re.sub(r"[\W_]+", "", t.lower())
        for store in KNOWN_STORES:
            store_clean = re.sub(r"[\W_]+", "", store)
            # If the pure text perfectly matches (e.g. "walmart" in "walmart")
            if store_clean in t_clean:
                score += 0.4
                break
            elif _char_sim(t_clean, store_clean) >= 0.8:
                score += 0.3
                break

        if t == t.upper() and alpha_ratio > 0.5:
            score += 0.2

        if re.match(r"^\d+", t):
            score -= 0.3
        if re.search(r"\b(tel|fax|gst|reg|no\.|@|www|\.com|bhd|sdn)\b", t, re.I):
            score -= 0.3
        if re.search(r"\d{5,}", t):
            score -= 0.2
        if re.search(r":\s*\S", t):
            score -= 0.25

        score += (1 - i / len(candidates)) * 0.1
        scored.append((score, t, line["confidence"]))

    if not scored:
        return _low_conf_field()

    scored.sort(key=lambda x: x[0], reverse=True)
    _, best_text, ocr_conf = scored[0]
    best_idx = next(i for i, (_, t, _) in enumerate(scored) if t == best_text)
    original_idx = next(i for i, line in enumerate(candidates) if line["text"].strip() == best_text)

    if len(best_text.split()) == 1 and original_idx + 1 < len(candidates):
        next_line = candidates[original_idx + 1]["text"].strip()
        next_alpha = _alpha_ratio(next_line)
        if next_alpha > 0.8 and len(next_line.split()) <= 2 and not re.search(r"\d", next_line):
            best_text = best_text + " " + next_line
    conf = round(min(1.0, max(0.0, scored[0][0])), 3)
    return _field(best_text, conf)


# ══════════════════════════════════════════════════════════════════════════════
# DATE
# ══════════════════════════════════════════════════════════════════════════════

def _extract_date(lines: list[dict]) -> dict:
    """
    Three-pass extraction:
      1. Lines with explicit "Date" label — most reliable
      2. Lines containing a timestamp (HH:MM) — date+time pairs are very reliable
      3. Any line matching a date pattern
    """
    # Pass 1 — explicit label
    for line in lines:
        if re.search(r"\bdate\b", line["text"], re.I):
            result = _match_date(line)
            if result:
                return result

    # Pass 2 — timestamp present
    for line in lines:
        if re.search(r"\d{2}:\d{2}", line["text"]):
            result = _match_date(line)
            if result:
                return result

    # Pass 3 — any valid date
    for line in lines:
        result = _match_date(line)
        if result:
            return result

    return _low_conf_field()


def _repair_date(text: str) -> str:
    """
    Fix common OCR corruptions in date strings.
    e.g. "04/2072016" → "04/20/2016"  (missing slash between day and year)
    e.g. "0420/2016"  → "04/20/2016"
    """
    # Pattern: digits / 4-6 digits (missing middle slash)
    # "04/2072016" — after first slash we have 7 digits: should be DD/YYYY
    repaired = re.sub(
        r"(\d{1,2})([\/\-\.])(\d{1,2})[7lI1\s]?(\d{4})\b",
        r"\1\2\3/\4",
        text
    )
    return repaired

def _match_date(line: dict) -> Optional[dict]:
    # Try original text first
    for pat in DATE_PATTERNS:
        m = pat.search(line["text"])
        if m and _valid_date(m.group(1)):
            conf = min(1.0, line["confidence"] * 0.85 + 0.15)
            return _field(m.group(1), round(conf, 3))

    # Try repaired text
    repaired = _repair_date(line["text"])
    if repaired != line["text"]:
        for pat in DATE_PATTERNS:
            m = pat.search(repaired)
            if m and _valid_date(m.group(1)):
                # Penalise confidence slightly — date was reconstructed
                conf = min(1.0, line["confidence"] * 0.75 + 0.15)
                return _field(m.group(1), round(conf, 3))

    return None


def _valid_date(value: str) -> bool:
    parts = re.split(r"[\/\-\.]", value)
    if len(parts) != 3:
        return False
    try:
        a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return False
    year = (c + 2000) if len(parts[2]) == 2 else c
    if not (2000 <= year <= 2099):
        return False
    if not ((1 <= a <= 31 and 1 <= b <= 12) or (1 <= b <= 31 and 1 <= a <= 12)):
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# TOTAL AMOUNT
# ══════════════════════════════════════════════════════════════════════════════

def _extract_total(lines: list[dict]) -> dict:
    """
    Priority ladder — tries each tier in order, returns on first match.
    
    Tier 1: Inclusive GST / Grand Total / Net Total  (most reliable)
    Tier 2: Plain TOTAL not blocked by skip list
    Tier 3: TOTAL SALES / TOTAL AMOUNT variants
    Tier 4: Spatial sweep — label on left, price on right
    """

    HARD_BLOCK = {
        "subtotal", "sub total", "sub-total",
        "total tax", "total gst", "total qty",
        "total sales excl", "total sales excluding", "total exclude",
        "excluded gst", "excl gst", "excl. gst",
        "discount", "rounding", "cash", "change",
        "cash tend", "debit tend", "visa tend",
        "tax 1", "tax 2", "tax 3",
        "total qty", "tot qty",
    }

    def is_blocked(t: str) -> bool:
        tl = t.lower()
        return any(b in tl for b in HARD_BLOCK)

    # ── Tier 1: Inclusive / Grand / Net ──────────────────────────────────────
    TIER1 = [
        "total sales inclusive",
        "total inclusive",
        "total incl",
        "inclusive gst",
        "grand total",
        "net total",
        "rounded total",
        "total amount",
        "total amt",
        "total (rm)",
        "total(rm)",
    ]
    for kw in TIER1:
        for line in lines:
            tl = line["text"].lower()
            if kw in tl and not is_blocked(tl):
                prices = PRICE_RE.findall(line["text"])
                if prices:
                    return _field(
                        prices[-1].replace(",", "."),
                        round(min(1.0, line["confidence"] * 0.85 + 0.15), 3)
                    )

    # ── Tier 2: Plain TOTAL ───────────────────────────────────────────────────
    # Must appear as a word boundary — avoids matching SUBTOTAL
    for line in lines:
        t  = line["text"]
        tl = t.lower()
        if re.search(r"\btotal\b", tl) and not is_blocked(tl):
            prices = PRICE_RE.findall(t)
            if prices:
                return _field(
                    prices[-1].replace(",", "."),
                    round(min(1.0, line["confidence"] * 0.85 + 0.15), 3)
                )

    subtotal_val = None
    tax_val      = None
    subtotal_conf = 0.0
    tax_conf      = 0.0

    for line in lines:
        tl = line["text"].lower()
        prices = PRICE_RE.findall(line["text"])

        if re.search(r"\bsubtotal\b|\bsub\s+total\b", tl):
            try:
                subtotal_val  = float(prices[-1].replace(",", ".")) if prices else None
                subtotal_conf = line["confidence"]
            except ValueError:
                pass

        elif re.search(r"\btax\b", tl) and not re.search(r"\btotal\b", tl):
            # Skip lines that show a tax RATE (contain %) — not a tax amount
            if "%" in line["text"]:
                # Extract the rate and compute tax from subtotal later
                rate_match = re.search(r"(\d+\.?\d*)\s*%", line["text"])
                if rate_match and subtotal_val is not None:
                    try:
                        rate    = float(rate_match.group(1)) / 100
                        tax_val = round(subtotal_val * rate, 2)
                        tax_conf = line["confidence"] * 0.75  # computed, penalise
                    except ValueError:
                        pass
                continue  # don't try price extraction on rate lines

        # Normal tax amount line (no %)
        if prices:
            try:
                tax_val  = float(prices[-1].replace(",", "."))
                tax_conf = line["confidence"]
            except ValueError:
                pass
        else:
            # Bare integer fallback — "49" → 0.49
            bare = re.search(r"\b(\d{1,3})\s*$", line["text"])
            if bare:
                try:
                    tax_val  = float(bare.group(1)) / 100
                    tax_conf = line["confidence"] * 0.7
                except ValueError:
                    pass

    if subtotal_val is not None and tax_val is not None:
        computed = round(subtotal_val + tax_val, 2)
        # Lower confidence — this is derived, not directly read
        conf = round(min(0.85, (subtotal_conf + tax_conf) / 2 * 0.8), 3)
        return _field(str(computed), conf)

    # If subtotal exists but no tax (tax = 0 or not shown)
    if subtotal_val is not None:
        conf = round(min(0.75, subtotal_conf * 0.8), 3)
        return _field(str(subtotal_val), conf)

    # ── Tier 3: TOTAL SALES / BAL ─────────────────────────────────────────────
    TIER3 = ["total sales", "bal", "balance due", "amount due"]
    for kw in TIER3:
        for line in lines:
            tl = line["text"].lower()
            if kw in tl and not is_blocked(tl):
                prices = PRICE_RE.findall(line["text"])
                if prices:
                    return _field(
                        prices[-1].replace(",", "."),
                        round(min(1.0, line["confidence"] * 0.85 + 0.15), 3)
                    )

    # ── Tier 4: Spatial sweep ─────────────────────────────────────────────────
    # Handles: "TOTAL :" label on left, price detached on right
    for line in lines:
        tl = line["text"].lower().strip()
        if re.fullmatch(r"total\s*:?", tl):
            anchor_y  = (line["bbox"][0][1] + line["bbox"][3][1]) / 2
            anchor_rx = max(line["bbox"][1][0], line["bbox"][2][0])
            for other in lines:
                if other is line:
                    continue
                oy = (other["bbox"][0][1] + other["bbox"][3][1]) / 2
                ox = min(other["bbox"][0][0], other["bbox"][3][0])
                if abs(oy - anchor_y) <= 15 and ox > anchor_rx - 20:
                    prices = PRICE_RE.findall(other["text"])
                    if prices:
                        return _field(
                            prices[-1].replace(",", "."),
                            round(min(1.0, other["confidence"] * 0.85 + 0.15), 3)
                        )

    return _low_conf_field()
# ══════════════════════════════════════════════════════════════════════════════
# ITEMS
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_prices(text: str) -> str:
    """
    Fix OCR decimal point corruptions:
    "3 99" → "3.99"  (space where decimal should be)
    "3,99" is already handled by PRICE_RE
    """
    # Pattern: single digit, space, exactly 2 digits, then space/end/letter
    return re.sub(r"\b(\d)\s(\d{2})\b", r"\1.\2", text)

def _extract_items(lines: list[dict]) -> list[dict]:
    """
    Detects layout type (LINEAR or TABULAR) then runs the appropriate
    multi-pattern item extraction:

    LINEAR  — item name and price on the same line
    TABULAR — price row first, item name on the line below (barcode receipts)
    SPLIT-2 — item name on one line, price on the next
    SPLIT-3 — item name / unit-price row / qty×final-price row
    """
    items   = []
    layout  = _detect_layout(lines)
    n       = len(lines)

    for i, line in enumerate(lines):
        t      = line["text"].strip()
        t     = _normalize_prices(t) 
        t = re.sub(r"(\d+)\s+(\d{2})\s*([A-Za-z]{0,2})$", r"\1.\2 \3", t)
        lower  = t.lower()

        if re.search(r"-\s*([A-Za-z]{0,2})?$", t):
            continue

        if PAYMENT_ZONE_RE.search(t):
            break

        
    

        if re.match(r"^\s*\d+\s*(ea|pc|lb|kg|x)\s", lower):
            continue
            
        # Explicitly skip scale/weight math lines on their own iteration
        if re.match(r"^\s*[\d.]+\s*(1b|lb|kg|oz|ea)\b", lower):
            continue
            
        if re.search(r"voided|void\s+entry|discount\s+given", lower):
            continue

        # Skip discount/coupon lines
        if re.search(r"\boff\b|\bcoupon\b|\bdiscount\b|\bsavings\b", lower):
            continue

        # Skip lines where price ends with "-" (negative/discount amount)
        if re.search(r"\d+\.\d+\s*[-—]", t):
            continue

        # Hard stop at financial summary zone
        if re.search(
            r"\b(total\s+(sales|amount|inclusive|exclusive|payable|tend|tax)"
            r"|net\s+total|grand\s+total|gst\s+summary"
            r"|tax\s+summary)\b",
            lower,
        ):
            
            break

        # Skip header/non-item lines
        if HEADER_SKIP_RE.search(t):
            continue
        if NON_ITEM_LINE_RE.search(t):
            math_stripped = re.sub(r"\d+\s*@\s*[\d.]+", "", t)
            math_stripped = re.sub(r"[\d.]+\s*(lb|1b|kg|oz|ea)\s*[@/]\s*[\d.]+", "", math_stripped, flags=re.I)

            # Try to pair with the PREVIOUS line as item name (weight line after item name)
            prices = PRICE_RE.findall(t)
            if prices and i > 0:
                prev_t     = lines[i - 1]["text"].strip()
                prev_alpha = _alpha_ratio(prev_t)
                prev_prices = PRICE_RE.findall(prev_t)
                if prev_alpha > 0.4 and not prev_prices and not NON_ITEM_LINE_RE.search(prev_t):
                    name = _clean_name(prev_t)
                    if len(name) >= 3:
                        price = prices[-1].replace(",", ".")
                        items.append(_item_entry(name, price, line["confidence"]))
            # If NO price is left after stripping the math, it is safe to skip.
            if not PRICE_RE.search(math_stripped):
                continue
        if ITEM_SKIP_RE.search(lower):
            continue
        if not re.search(r"[A-Za-z]", t) :
            continue

        prices = PRICE_RE.findall(t)

        # ── SPLIT-3: name → unit-price row → qty×final-price row ─────────────
        if not prices and i + 2 < n:
            next_t  = lines[i + 1]["text"].strip()
            next2_t = lines[i + 2]["text"].strip()
            next2_prices = PRICE_RE.findall(next2_t)

            if (not PRICE_RE.findall(next_t)       # line+1 has no price
                    and next2_prices                # line+2 has the price
                    and QTY_LINE_RE.search(next2_t) # line+2 looks like qty row
                    and _alpha_ratio(t) > 0.3):
                name = _clean_name(t)
                if len(name) >= 3:
                    price = next2_prices[-1].replace(",", ".")
                    items.append(_item_entry(name, price, line["confidence"]))
                continue

        # ── SPLIT-2: name only → price on next line ───────────────────────────
        if not prices and i + 1 < len(lines):
            next_t = lines[i+1]["text"].strip()
            
            # Repair missing decimals in next_t before checking!
            next_t = re.sub(r"(\d+)\s+(\d{2})\s*([A-Za-z]{0,2})$", r"\1.\2 \3", next_t)
            next_prices = re.findall(PRICE_RE, next_t)
            
            has_alpha = len(re.findall(r"[A-Za-z]", t)) > 2
            
            # Robust math-line detection (replaces brittle fullmatch)
            # Allow up to 6 letters (lb, kg, N). Reject if there is any 4+ letter word.
            letter_count = len(re.findall(r"[A-Za-z]", next_t))
            has_long_word = bool(re.search(r"[A-Za-z]{4,}", next_t))
            next_is_math_line = (letter_count <= 6) and not has_long_word

            if has_alpha and next_prices and next_is_math_line \
               and not NON_ITEM_LINE_RE.search(t) \
               and not HEADER_SKIP_RE.search(t):
                
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

        # ── TABULAR: price row first, name below ──────────────────────────────
        m = ITEM_PRICE_RE.match(t)
        name = m.group(1).strip() if m else ""

        if _alpha_ratio(name) < 0.2 and layout == "TABULAR" and i + 1 < n:
            next_t     = lines[i + 1]["text"].strip()
            next_alpha = _alpha_ratio(next_t)
            if next_alpha > 0.4 and not PRICE_RE.findall(next_t):
                name = next_t

        # ── LINEAR: price on same line ────────────────────────────────────────
        elif _alpha_ratio(name) < 0.2 and layout == "LINEAR" and i > 0:
            prev_t     = lines[i - 1]["text"].strip()
            prev_alpha = _alpha_ratio(prev_t)
            if prev_alpha > 0.4 and not PRICE_RE.findall(prev_t):
                name = prev_t

        name = _clean_name(name)
        if len(name) < 3:
            continue
        if len(re.findall(r"\d", name)) / max(len(name), 1) > 0.6:
            continue

        price = prices[-1].replace(",", ".")
        items.append(_item_entry(name, price, line["confidence"]))

    # Deduplicate identical name+price pairs
    seen, deduped = set(), []
    for entry in items:
        key = (entry["name"].lower().strip(), entry["price"])
        if key not in seen:
            seen.add(key)
            deduped.append(entry)

    return deduped


def _detect_layout(lines: list[dict]) -> str:
    """
    Classify receipt as LINEAR or TABULAR by checking whether
    the first priced line is followed by an alphabetic-only line.
    """
    for i, line in enumerate(lines):
        if PRICE_RE.findall(line["text"]):
            if i + 1 < len(lines):
                next_t = lines[i + 1]["text"]
                if _alpha_ratio(next_t) > 0.5 and not PRICE_RE.findall(next_t):
                    return "TABULAR"
            return "LINEAR"
    return "LINEAR"


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _clean_name(name: str) -> str:
    name = re.sub(r"\s+\d{5,}\s*[A-Z]?\s*$", "", name)   # trailing barcode
    name = re.sub(r"^[A-Z]\s{2,}\d{5,}\s+", "", name)    # ← NEW: "E    673919 "
    name = re.sub(r"^\d{5,}\s+", "", name)                # ← leading barcode only
    name = re.sub(r"[©®™\*#]", "", name)
    name = re.sub(r"^\d+\s+", "", name)
    name = re.sub(r"^[\W_]+", "", name)
    name = re.sub(r"\s+\d{5,}\s*[A-Z]?\s*$", "", name)
    return name.strip()


def _item_entry(name: str, price: str, ocr_conf: float) -> dict:
    conf = round(min(1.0, ocr_conf * 0.9 + 0.05), 3)
    entry = {"name": name, "price": price, "confidence": conf}
    if conf < 0.7:
        entry["flag"] = "LOW_CONFIDENCE"
    return entry


def _field(value, confidence: float) -> dict:
    entry = {"value": value, "confidence": confidence}
    if confidence < 0.7:
        entry["flag"] = "LOW_CONFIDENCE"
    return entry


def _low_conf_field() -> dict:
    return {"value": None, "confidence": 0.0, "flag": "LOW_CONFIDENCE"}


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(re.findall(r"[A-Za-z]", text)) / len(text)


def _char_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    matches = sum(ca == cb for ca, cb in zip(a, b))
    return matches / max(len(a), len(b))
