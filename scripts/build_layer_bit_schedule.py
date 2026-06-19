#!/usr/bin/env python3
"""Build a layer-wise bit schedule from attention error probe output."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_json")
    parser.add_argument("--method", default="turboquant_2_5bit")
    parser.add_argument("--metric", default="score_rmse")
    parser.add_argument("--target-average-bits", type=float, default=2.5)
    parser.add_argument("--low-bits", type=float, default=2.0)
    parser.add_argument("--high-bits", type=float, default=3.0)
    parser.add_argument("--output", default="reproduce/incremental/layer_schedule_from_attention_error.json")
    args = parser.parse_args()

    data = json.loads(Path(args.probe_json).read_text(encoding="utf-8"))
    by_layer = defaultdict(list)
    for row in data["summary"]:
        if row["method"] != args.method:
            continue
        value = row.get(args.metric)
        if value is not None:
            by_layer[int(row["layer"])].append(float(value))
    if not by_layer:
        raise ValueError(f"no rows found for method={args.method!r} metric={args.metric!r}")

    layer_scores = [
        {"layer": layer, "score": sum(values) / len(values), "num_heads": len(values)}
        for layer, values in sorted(by_layer.items())
    ]
    num_layers = len(layer_scores)
    high_fraction = (args.target_average_bits - args.low_bits) / (args.high_bits - args.low_bits)
    high_count = round(num_layers * high_fraction)
    high_count = max(0, min(num_layers, high_count))
    high_layers = {
        row["layer"]
        for row in sorted(layer_scores, key=lambda item: (-item["score"], item["layer"]))[:high_count]
    }
    schedule = [args.high_bits if layer in high_layers else args.low_bits for layer in range(num_layers)]
    average_bits = sum(schedule) / len(schedule)
    output = {
        "source": str(Path(args.probe_json)),
        "method": args.method,
        "metric": args.metric,
        "target_average_bits": args.target_average_bits,
        "low_bits": args.low_bits,
        "high_bits": args.high_bits,
        "num_layers": num_layers,
        "high_count": high_count,
        "average_bits": average_bits,
        "layer_scores": layer_scores,
        "high_layers": sorted(high_layers),
        "layer_key_bits": schedule,
        "layer_value_bits": schedule,
        "cli": {
            "layer_key_bits": ",".join(f"{value:g}" for value in schedule),
            "layer_value_bits": ",".join(f"{value:g}" for value in schedule),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "average_bits": average_bits, "high_layers": sorted(high_layers)}, indent=2))


if __name__ == "__main__":
    main()
