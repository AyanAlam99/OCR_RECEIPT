import sys
import json
from pathlib import Path
from src.preprocesor import preprocess
from src.ocr          import run_ocr
from src.extractor    import extract_fields
from src.summarizer   import generate_summary
from src.confidence import score_fields
import numpy as np


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def demo_single(image_path: str) -> dict:
    print(f"\nProcessing: {image_path}")

    img   = preprocess(image_path)
    lines = run_ocr(img)

    print("\n" + "=" * 52)
    print(f"{'  OCR RECONSTRUCTION  ':=^52}")
    print("=" * 52)
    for line in lines:
        print(f"  [{line['confidence']:.2f}]  {line['text']}")
    print("=" * 52)

    fields   = extract_fields(lines)
    fields   = score_fields(fields, lines)   
    avg_conf = round(float(np.mean([l["confidence"] for l in lines])), 3) if lines else 0.0

    print(f"\n  Store  : {fields['store_name']}")
    print(f"  Date   : {fields['date']}")
    print(f"  Total  : {fields['total_amount']}")
    print(f"  Items  ({len(fields['items'])}):")
    for item in fields["items"]:
        flag = " ⚠️" if item.get("flag") else ""
        print(f"    {item['name']:<35} {item['price']:>8}  [{item['confidence']:.2f}]{flag}")

    return {
        "source":             Path(image_path).name,
        "avg_ocr_confidence": avg_conf,
        **fields,
    }
def main():
    paths = sys.argv[1:]

    if not paths:
        print("Usage: python demo.py <image_or_folder> [image2 ...]")
        sys.exit(1)

    expanded = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            expanded += sorted(
                str(f) for f in path.iterdir()
                if f.suffix.lower() in SUPPORTED_EXTS
            )
        else:
            expanded.append(p)

    results = [demo_single(p) for p in expanded]

    summary = generate_summary(results)
    print("\n FINANCIAL SUMMARY")
    print(json.dumps(summary, indent=2))

    out_dir = Path("outputs2")
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        if "error" in r:
            continue
            
        name = Path(r.get("source", "receipt")).stem
        
        deliverable = {
            "store_name": r["store_name"],
            "date": r["date"],
            "items": r["items"],
            "total_amount": r["total_amount"]
        }
        
        out_path = out_dir / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(deliverable, f, indent=2)

    summary_path = out_dir / "financial_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n Outputs saved to {out_dir.resolve()}/")

if __name__ == "__main__":
    main()