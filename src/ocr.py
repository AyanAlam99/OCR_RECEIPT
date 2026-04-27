import re
import os
import tempfile
import cv2
import numpy as np
from paddleocr import PaddleOCR

_engine: PaddleOCR | None = None


def get_engine() -> PaddleOCR:
    global _engine
    if _engine is None:
        _engine = PaddleOCR(use_angle_cls=True, lang="en")
    return _engine




def run_ocr(img: np.ndarray) -> list[dict]:
 
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)
    cv2.imwrite(tmp_path, img)

    try:
        raw = get_engine().ocr(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    detections = _parse_raw(raw)
    return _merge_rows(detections)


def _parse_raw(raw) -> list[dict]:
    """Extract text / score / bbox from PaddleOCR's nested output."""
    lines = []
    if not raw:
        return lines
    for page in raw:
        if not isinstance(page, dict):
            continue
        texts  = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        polys  = page.get("rec_polys", [])
        for text, score, bbox in zip(texts, scores, polys):
            text = text.strip()
            if text:
                lines.append({
                    "text":       text,
                    "confidence": round(float(score), 4),
                    "bbox":       bbox.tolist(),
                })
    return lines


def _top_y(line: dict) -> float:
    return min(pt[1] for pt in line["bbox"])


def _left_x(line: dict) -> float:
    return min(pt[0] for pt in line["bbox"])


def _right_x(line: dict) -> float:
    return max(pt[0] for pt in line["bbox"])


def _merge_rows(lines: list[dict], y_tolerance: int = 10) -> list[dict]:
    if not lines:
        return lines

    sorted_lines = sorted(lines, key=_top_y)

    # Group into rows
    rows: list[list[dict]] = []
    current_row = [sorted_lines[0]]

    for line in sorted_lines[1:]:
        if abs(_top_y(line) - _top_y(current_row[0])) <= y_tolerance:
            current_row.append(line)
        else:
            rows.append(current_row)
            current_row = [line]
    rows.append(current_row)

    # Merge each row into one text string
    merged = []
    for row in rows:
        row = sorted(row, key=_left_x)
        text   = ""
        prev_rx = None

        for det in row:
            lx = _left_x(det)
            rx = _right_x(det)
            if prev_rx is not None:
                gap    = lx - prev_rx
                spaces = max(1, int(gap / 12))  
                text  += " " * spaces
            text    += det["text"]
            prev_rx  = rx

        avg_conf = round(sum(d["confidence"] for d in row) / len(row), 4)
        merged.append({
            "text":       text,
            "confidence": avg_conf,
            "bbox":       row[0]["bbox"],   # leftmost anchor
        })

    return merged
