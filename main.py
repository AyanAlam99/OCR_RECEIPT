import sys
import json
import argparse
from pathlib import Path
from pipeline import process_image, process_batch


def main():
    parser = argparse.ArgumentParser(description="Receipt OCR Pipeline")
    parser.add_argument("images", nargs="*",
                        help="One or more image paths (single-image mode)")
    parser.add_argument("--batch",  metavar="DIR",
                        help="Folder of images to process in batch mode")
    parser.add_argument("--output", metavar="DIR", default="outputs",
                        help="Output folder (default: outputs/)")
    args = parser.parse_args()

    if args.batch:
        process_batch(args.batch, args.output)
        return
    if not args.images:
        default_dir = Path(__file__).parent / "images"
        if default_dir.exists():
            process_batch(str(default_dir), args.output)
        else:
            print("Usage: python main.py <image> [image2 ...]")
            print(" python main.py --batch <folder>")
            sys.exit(1)
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in args.images:
        print(f"\nProcessing: {img_path}")
        try:
            result = process_image(img_path)
        except Exception as e:
            print(f"  {e}")
            continue

        print(f"  Store  : {result['store_name']['value']}")
        print(f"  Date   : {result['date']['value']}")
        print(f"  Items  : {len(result['items'])}")
        print(f"  Total  : {result['total_amount']['value']}")

        out_path = out_dir / (Path(img_path).stem + ".json")
        deliverable = {
            "store_name": result["store_name"],
            "date": result["date"],
            "items": result["items"],
            "total_amount": result["total_amount"]
        }

        with open(out_path, "w") as f:
            json.dump(deliverable, f, indent=2)
        print(f" -> Saved: {out_path}")


if __name__ == "__main__":
    main()
