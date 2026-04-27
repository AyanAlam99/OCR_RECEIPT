import re
from typing import Optional


W_STORE  = {"ocr": 0.5, "pattern": 0.3, "anchor": 0.2}
W_DATE   = {"ocr": 0.4, "pattern": 0.4, "anchor": 0.2}
W_TOTAL  = {"ocr": 0.4, "pattern": 0.3, "anchor": 0.3}
W_ITEMS  = {"ocr": 0.6, "pattern": 0.3, "reconciliation": 0.1}

LOW_CONF_THRESHOLD = 0.7

def score_fields(fields: dict, lines: list[dict]) -> dict:
    scored = {}

    scored["store_name"]   = _score_store(fields["store_name"], lines)
    scored["date"]         = _score_date(fields["date"], lines)
    scored["total_amount"] = _score_total(fields["total_amount"], lines)
    scored["items"]        = _score_items(
                                fields["items"],
                                fields["total_amount"].get("value")
                             )

    scored = _reconcile(scored,lines)

    return scored

KNOWN_STORES = {
    "walmart", "walgreens", "target", "costco", "kroger",
    "whole foods", "trader joe", "cvs", "aldi", "safeway",
    "publix", "meijer", "heb", "lidl", "dollar tree", "spar",
    "brewery tap", "asia mart",
}


def _score_store(field: dict, lines: list[dict]) -> dict:
    if not field.get("value"):
        return _low(field)

    value    = field["value"]
    ocr_conf = field.get("confidence", 0.5)

    # Pattern validity ,does it look like a store name?
    alpha_ratio = len(re.findall(r"[A-Za-z]", value)) / max(len(value), 1)
    has_digits_only = bool(re.fullmatch(r"[\d\s]+", value))
    is_known_store  = any(s in value.lower() for s in KNOWN_STORES)

    pattern_score = 0.0
    if has_digits_only:
        pattern_score = 0.0
    elif is_known_store:
        pattern_score = 1.0
    elif alpha_ratio > 0.7 and len(value) >= 4:
        pattern_score = 0.7
    elif alpha_ratio > 0.5:
        pattern_score = 0.5
    else:
        pattern_score = 0.2

    # Anchor proximity , is it in the top 15% of the receipt?
    anchor_score = _position_score(value, lines, top_fraction=0.15)

    composite = (
        W_STORE["ocr"]     * ocr_conf +
        W_STORE["pattern"] * pattern_score +
        W_STORE["anchor"]  * anchor_score
    )

    return _make_field(value, composite)

DATE_RE = re.compile(
    r"^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$"
    r"|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}"
    r"|[A-Za-z]+ \d{1,2},?\s*\d{4}"
)


def _score_date(field: dict, lines: list[dict]) -> dict:
    if not field.get("value"):
        return _low(field)

    value    = field["value"]
    ocr_conf = field.get("confidence", 0.5)

    # Pattern validity, does it match a known date format?
    pattern_score = 1.0 if DATE_RE.match(value.strip()) else 0.3

    # Extra boost if value is a plausible calendar date
    parts = re.split(r"[\/\-\.]", value)
    if len(parts) == 3:
        try:
            nums = [int(p) for p in parts]
            # Check at least one part looks like a year
            if any(2000 <= n <= 2099 for n in nums):
                pattern_score = min(1.0, pattern_score + 0.1)
        except ValueError:
            pass

    # Anchor proximity, date label found on same line?
    anchor_score = _label_found_score(value, lines, labels=["date", "date:"])

    composite = (
        W_DATE["ocr"]     * ocr_conf +
        W_DATE["pattern"] * pattern_score +
        W_DATE["anchor"]  * anchor_score
    )

    return _make_field(value, composite)

CURRENCY_RE = re.compile(r"^\d{1,6}(\.\d{2})?$")

STRONG_TOTAL_KEYWORDS = {
    "grand total", "net total", "total sales inclusive",
    "total amount", "total (rm)", "total incl",
}
WEAK_TOTAL_KEYWORDS = {"total", "bal", "amount due"}


def _score_total(field: dict, lines: list[dict]) -> dict:
    if not field.get("value"):
        return _low(field)

    value    = field["value"]
    ocr_conf = field.get("confidence", 0.5)

    # Pattern validity, is it a valid currency string?
    clean = value.replace(",", "").strip()
    pattern_score = 1.0 if CURRENCY_RE.match(clean) else 0.2

    # Anchor proximity , which keyword tier matched?
    anchor_score = 0.0
    for line in lines:
        tl = line["text"].lower()
        if any(kw in tl for kw in STRONG_TOTAL_KEYWORDS):
            prices = re.findall(r"\d{1,4}[.,]\d{2}", line["text"])
            if prices and prices[-1].replace(",", ".") == clean:
                anchor_score = 1.0
                break
        if any(kw in tl for kw in WEAK_TOTAL_KEYWORDS):
            prices = re.findall(r"\d{1,4}[.,]\d{2}", line["text"])
            if prices and prices[-1].replace(",", ".") == clean:
                anchor_score = 0.7

    composite = (
        W_TOTAL["ocr"]     * ocr_conf +
        W_TOTAL["pattern"] * pattern_score +
        W_TOTAL["anchor"]  * anchor_score
    )

    return _make_field(value, composite)

def _score_items(items: list[dict], total_value: Optional[str]) -> list[dict]:
    if not items:
        return items

    scored_items = []
    for item in items:
        ocr_conf = item.get("confidence", 0.5)

        # Pattern validity ,is the price a valid currency value?
        price_str = item.get("price", "")
        price_valid = bool(re.match(r"^\d{1,6}(\.\d{2,3})?$",
                                     price_str.replace(",", "")))
        pattern_score = 1.0 if price_valid else 0.3

        # Name validity, is the name alphabetic enough to be real?
        name = item.get("name", "")
        alpha_ratio = len(re.findall(r"[A-Za-z]", name)) / max(len(name), 1)
        name_score = 1.0 if alpha_ratio > 0.3 else 0.5

        composite = (
            W_ITEMS["ocr"]     * ocr_conf +
            W_ITEMS["pattern"] * ((pattern_score + name_score) / 2)
        )

        entry = {
            "name":       name,
            "price":      price_str,
            "confidence": round(min(1.0, composite), 3),
        }
        if entry["confidence"] < LOW_CONF_THRESHOLD:
            entry["flag"] = "LOW_CONFIDENCE"
        scored_items.append(entry)

    return scored_items

def _reconcile(scored: dict, lines: list[dict]) -> dict:
    items       = scored.get("items", [])
    total_field = scored.get("total_amount", {})
    total_value = total_field.get("value")

    if not items or not total_value:
        return scored

    # Sum item prices
    item_sum = 0.0
    for item in items:
        try:
            item_sum += float(item["price"].replace(",", "."))
        except (ValueError, KeyError):
            pass

    try:
        total_float = float(total_value.replace(",", "."))
    except ValueError:
        return scored

    # the explicit Subtotal in the raw lines
    subtotal_float = None
    for line in lines:
        if re.search(r"\bsubtotal\b|\bsub\s+total\b", line["text"].lower()):
            prices = re.findall(r"\d{1,4}[.,]\d{2}", line["text"])
            if prices:
                subtotal_float = float(prices[-1].replace(",", "."))
                break

    difference_to_subtotal = abs(item_sum - subtotal_float) if subtotal_float else None
    difference_to_total    = abs(item_sum - total_float)
    
    # Tiered Validation Logic
    items_match = False
    status_msg = "MISMATCH"
    
    # Exact match with Subtotal 
    if difference_to_subtotal is not None and difference_to_subtotal < 0.05:
        items_match = True
        status_msg = "MATCH_SUBTOTAL"
        
    # Match with Grand Total (Allowing up to 15% for implicit taxes)
    elif difference_to_total <= max(total_float * 0.15, 0.10):
        items_match = True
        status_msg = "MATCH_TOTAL_WITH_TAX"

    if not items_match:
        # Calculate penalty based on how far off we are from the Grand Total
        penalty = min(0.4, difference_to_total / max(total_float, 1.0))

        # Penalize total confidence
        current_total_conf = total_field.get("confidence", 0.5)
        new_total_conf = round(max(0.1, current_total_conf - penalty), 3)
        scored["total_amount"] = _make_field(total_value, new_total_conf)
        
        # Penalize each item's confidence
        penalized_items = []
        for item in items:
            new_item_conf = round(max(0.1, item.get("confidence", 0.5) - (penalty / 2)), 3)
            new_item = {**item, "confidence": new_item_conf}
            if new_item_conf < LOW_CONF_THRESHOLD:
                new_item["flag"] = "LOW_CONFIDENCE"
            penalized_items.append(new_item)
        scored["items"] = penalized_items


    scored["total_amount"]["reconciliation"] = {
        "item_sum":  round(item_sum, 2),
        "subtotal":  subtotal_float,
        "total":     total_float,
        "status":    status_msg
    }

    return scored

def _position_score(value: str, lines: list[dict], top_fraction: float = 0.15) -> float:
    """
    Returns 1.0 if the value appears in the top `top_fraction` of the receipt,
    0.5 if in top half, 0.2 otherwise.
    """
    if not lines:
        return 0.5

    all_y      = [min(pt[1] for pt in l["bbox"]) for l in lines]
    min_y      = min(all_y)
    max_y      = max(all_y)
    height     = max(max_y - min_y, 1)
    threshold  = min_y + height * top_fraction

    for line in lines:
        if value.lower() in line["text"].lower():
            line_y = min(pt[1] for pt in line["bbox"])
            if line_y <= threshold:
                return 1.0
            elif line_y <= min_y + height * 0.5:
                return 0.5
            else:
                return 0.2
    return 0.3


def _label_found_score(value: str, lines: list[dict], labels: list[str]) -> float:
    """
    Returns 1.0 if the value appears on a line that also contains one of the labels.
    Returns 0.5 if value found but no label on same line.
    Returns 0.2 if value not found at all.
    """
    for line in lines:
        tl = line["text"].lower()
        if value.lower() in tl:
            if any(lbl in tl for lbl in labels):
                return 1.0
            return 0.5
    return 0.2


def _make_field(value, confidence: float) -> dict:
    confidence = round(min(1.0, max(0.0, confidence)), 3)
    entry = {"value": value, "confidence": confidence}
    if confidence < LOW_CONF_THRESHOLD:
        entry["flag"] = "LOW_CONFIDENCE"
    return entry


def _low(field: dict) -> dict:
    return {"value": field.get("value"), "confidence": 0.0, "flag": "LOW_CONFIDENCE"}