def generate_summary(results: list[dict]) -> dict:
    total_spend:    float       = 0.0
    transactions:   int         = 0
    spend_per_store: dict       = {}
    failed:          list[str]  = []

    for r in results:
        if "error" in r:
            failed.append(r.get("source", "unknown"))
            continue

        transactions += 1
        store   = r.get("store_name", {}).get("value") or "Unknown"
        amt_raw = r.get("total_amount", {}).get("value")

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
        "total_spend":       round(total_spend, 2),
        "spend_per_store":   spend_per_store,
        "failed_images":     failed,
    }
