import json
import argparse
import re
from pathlib import Path


def _normalize_price(value: str) -> float:
    try:
        return round(float(re.sub(r"[^\d.]", "", value)), 2)
    except (ValueError, TypeError):
        return -1.0


def _normalize_text(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"[^\w\s]", "", value.lower()).strip()


def _store_match(predicted: str, ground_truth: str) -> bool:
    if not predicted or not ground_truth:
        return False
    p = _normalize_text(predicted)
    g = _normalize_text(ground_truth)
    return p in g or g in p or p == g


def _date_match(predicted: str, ground_truth: str) -> bool:

    if not predicted or not ground_truth:
        return False
    p_nums = re.findall(r"\d+", predicted)
    g_nums = re.findall(r"\d+", ground_truth)
    return p_nums == g_nums


def _total_match(predicted: str, ground_truth: str, tolerance: float = 0.02) -> bool:
   
    p = _normalize_price(predicted)
    g = _normalize_price(ground_truth)
    if p < 0 or g < 0:
        return False
    return abs(p - g) <= tolerance


def _items_match(predicted: list, ground_truth: list) -> dict:

    if not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched = 0
    gt_names = [_normalize_text(i.get("name", "")) for i in ground_truth]
    gt_prices = [_normalize_price(i.get("price", "")) for i in ground_truth]

    for pred_item in predicted:
        pred_name  = _normalize_text(pred_item.get("name", ""))
        pred_price = _normalize_price(pred_item.get("price", ""))

        for j, (gt_name, gt_price) in enumerate(zip(gt_names, gt_prices)):
            name_ok  = pred_name in gt_name or gt_name in pred_name
            price_ok = abs(pred_price - gt_price) <= 0.02 if pred_price >= 0 else False
            if name_ok and price_ok:
                matched += 1
                break

    precision = matched / len(predicted)
    recall    = matched / len(ground_truth)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
    }

def evaluate_against_ground_truth(gt_dir: str, output_dir: str) -> dict:
    gt_path  = Path(gt_dir)
    out_path = Path(output_dir)

    results = []
    gt_files = list(gt_path.glob("*.json"))

    if not gt_files:
        print(f"No ground truth files found in '{gt_dir}'")
        return {}

    for gt_file in sorted(gt_files):
        stem     = gt_file.stem
        out_file = out_path / f"{stem}.json"

        if not out_file.exists():
            print(f"  [SKIP] No output found for {stem}")
            continue

        with open(gt_file)  as f: gt   = json.load(f)
        with open(out_file) as f: pred = json.load(f)

        pred_store = pred.get("store_name",   {}).get("value", "")
        pred_date  = pred.get("date",         {}).get("value", "")
        pred_total = pred.get("total_amount", {}).get("value", "")
        pred_items = pred.get("items", [])

        r = {
            "file":          stem,
            "store_correct": _store_match(pred_store, gt.get("store_name", "")),
            "date_correct":  _date_match(pred_date,   gt.get("date",       "")),
            "total_correct": _total_match(pred_total, gt.get("total_amount", "")),
            "items":         _items_match(pred_items, gt.get("items", [])),
            "predicted": {
                "store": pred_store,
                "date":  pred_date,
                "total": pred_total,
                "item_count": len(pred_items),
            },
            "ground_truth": {
                "store": gt.get("store_name"),
                "date":  gt.get("date"),
                "total": gt.get("total_amount"),
                "item_count": len(gt.get("items", [])),
            }
        }
        results.append(r)

    if not results:
        return {}

    n = len(results)
    summary = {
        "total_evaluated":   n,
        "store_accuracy":    round(sum(r["store_correct"] for r in results) / n, 3),
        "date_accuracy":     round(sum(r["date_correct"]  for r in results) / n, 3),
        "total_accuracy":    round(sum(r["total_correct"] for r in results) / n, 3),
        "items_avg_f1":      round(sum(r["items"]["f1"]   for r in results) / n, 3),
        "items_avg_precision": round(sum(r["items"]["precision"] for r in results) / n, 3),
        "items_avg_recall":    round(sum(r["items"]["recall"]    for r in results) / n, 3),
        "per_file": results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR Pipeline")
    parser.add_argument("--ground-truth", metavar="DIR",
                        help="Folder containing ground truth JSON files")
    parser.add_argument("--outputs", metavar="DIR", default="outputs",
                        help="Folder containing pipeline JSON outputs")
    parser.add_argument("--save", metavar="FILE", default="evaluation_report.json",
                        help="Save evaluation report to this file")
    args = parser.parse_args()

    report = {}

    if args.ground_truth:
        print("\n GROUND TRUTH EVALUATION")
        gt_results = evaluate_against_ground_truth(args.ground_truth, args.outputs)
        report["ground_truth"] = gt_results
        if gt_results:
            print(f"  Store accuracy  : {gt_results['store_accuracy']}")
            print(f"  Date accuracy   : {gt_results['date_accuracy']}")
            print(f"  Total accuracy  : {gt_results['total_accuracy']}")
            print(f"  Items F1        : {gt_results['items_avg_f1']}")

    with open(args.save, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n Report saved ->  {args.save}")


if __name__ == "__main__":
    main()