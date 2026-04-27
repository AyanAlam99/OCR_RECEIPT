# OCR Receipt Pipeline

A system that extracts structured information from receipt images using PaddleOCR, with confidence-aware outputs and financial summarization.

## What it does

- Extracts store name, date, items, and total amount from receipt images
- Assigns composite confidence scores to each extracted field
- Flags low-confidence extractions (< 0.7)
- Reconciles item prices against total amount
- Generates a financial summary across all processed receipts

## Project Structure

```
OCR_RECEIPT/
├── src/
│   ├── preprocesor.py     # Image preprocessing (CLAHE, denoise, deskew)
│   ├── ocr.py             # PaddleOCR wrapper + spatial row merging
│   ├── extractor.py       # Field extraction (store, date, items, total)
│   ├── confidence.py      # Composite confidence scoring + reconciliation
│   └── summarizer.py      # Financial aggregation across receipts
├── data/
│   ├── raw/               # Input receipt images
│   └── ground_truth/      # Manually annotated JSONs for evaluation
├── outputs/               # Per-receipt JSON outputs + financial summary
├── pipeline.py            # Orchestrator tying all phases together
├── main.py                # CLI entry point (batch or single image)
├── demo_single_run.py     # Debug tool — prints OCR reconstruction
├── evaluation.py          # Evaluation script (ground truth + heuristic)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

First run will download PaddleOCR models (~200MB).

## Usage

**Single image:**
```bash
python main.py receipt.jpg
```

**Batch folder:**
```bash
python main.py --batch data/raw/ --output outputs/
```

**Debug a single image (shows OCR reconstruction):**
```bash
python demo_single_run.py receipt.jpg
```

**Run evaluation:**
```bash
python evaluation.py --ground-truth data/ground_truth/ --outputs outputs/
```

## Output Format

Each receipt produces a JSON file:

```json
{
  "store_name": { "value": "TRADER JOE'S", "confidence": 0.972 },
  "date":       { "value": "06-28-2014",   "confidence": 0.897 },
  "items": [
    { "name": "R-CARROTS SHREDDED 10 OZ", "price": "1.29", "confidence": 0.86 }
  ],
  "total_amount": {
    "value": "38.68",
    "confidence": 0.91,
    "reconciliation": {
      "item_sum": 38.68,
      "subtotal": 38.68,
      "total": 38.68,
      "status": "MATCH_SUBTOTAL"
    }
  }
}
```

Fields with confidence below 0.7 are flagged with `"flag": "LOW_CONFIDENCE"`.

## Confidence Scoring

Each field score is computed as a weighted composite:

```
Field_Score = (w1 × OCR_logit) + (w2 × Pattern_Validity) + (w3 × Anchor_Proximity)
```

After extraction, a reconciliation check sums all item prices and compares against the extracted total. A mismatch penalizes both item and total confidence scores.

## Evaluation Results (19 receipts)

| Metric | Score |
|--------|-------|
| Store accuracy | 73.7% |
| Date accuracy | 84.2% |
| Total accuracy | 73.7% |
| Items F1 | 42.7% |

## Known Limitations

- Hough transform deskew fails on rotations > 45°
- Item extraction accuracy drops on heavily photographed (curved/blurry) receipts
- Receipts with no standard total keyword fall back to mathematical derivation (lower confidence)
- Items F1 is lower because tax lines and weight-based items are intentionally excluded
