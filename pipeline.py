import json
import numpy as np
from pathlib import Path
from src.preprocesor import preprocess
from src.ocr import run_ocr
from src.extractor   import extract_fields
from src.confidence import score_fields
from src.summarizer   import generate_summary


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def process_image(image_path: str) -> dict:

    img   = preprocess(image_path)

    lines = run_ocr(img)
    
    if not lines:
        return {
            "source": Path(image_path).name,
            "avg_ocr_confidence": 0.0,
            "store_name":   {"value": None, "confidence": 0.0, "flag": "LOW_CONFIDENCE"},
            "date":         {"value": None, "confidence": 0.0, "flag": "LOW_CONFIDENCE"},
            "items":        [],
            "total_amount": {"value": None, "confidence": 0.0, "flag": "LOW_CONFIDENCE"},
        }

    fields   = extract_fields(lines)
    fields   = score_fields(fields, lines) 
    avg_conf = round(float(np.mean([l["confidence"] for l in lines])), 3)

    return {
        "source":             Path(image_path).name,
        "avg_ocr_confidence": avg_conf,
        **fields,
    }


def process_batch(image_dir: str, output_dir: str = "outputs") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )

    if not image_paths:
        print(f"No images found in '{image_dir}'.")
        return
    print(f"  Receipt OCR Pipeline  —> {len(image_paths)} image(s)")

    all_results = []

    for path in image_paths:
        print(f"\n {path.name}")
        try:
            result = process_image(str(path))
        except Exception as e:
            print(f" {e}")
            all_results.append({"error": str(e), "source": path.name})
            continue

        all_results.append(result)
        out_path = out / (path.stem + ".json")
        deliverable = {
        "store_name":   result["store_name"],
        "date":         result["date"],
        "items":        result["items"],
        "total_amount": result["total_amount"]
    }
        with open(out_path, "w") as f:
            json.dump(deliverable, f, indent=2)

        print(f" store  : {result['store_name']['value']}")
        print(f" date   : {result['date']['value']}")
        print(f" items  : {len(result['items'])}")
        print(f" total  : {result['total_amount']['value']}")
        print(f" -> {out_path}")

    summary      = generate_summary(all_results)
    summary_path = out / "financial_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*52}")
    print(f"  Transactions : {summary['num_transactions']}")
    print(f"  Total spend  : {summary['total_spend']}")
    print(f"  Summary      → {summary_path}")
    print(f"{'='*52}\n")
