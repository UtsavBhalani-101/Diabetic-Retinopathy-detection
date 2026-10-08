"""Run the standalone TTECAM targeted-vs-random occlusion experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from ttecam_experiment.common import (
    add_common_args,
    build_idrid_test_dataset,
    build_loader,
    evaluate_condition_deterministic,
    initialize,
    load_model,
    load_temperature,
    project_root,
    write_results_json,
    write_summary_csv,
)
from ttecam_experiment.occlusion_datasets import (
    HeatmapOccludedDataset,
    RandomMeanOccludedDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--heatmap-dir",
        default="artifacts/ttecam_heatmaps/idrid/test",
        help="Directory containing TTECAM {idx}.npy heatmaps.",
    )
    parser.add_argument(
        "--top-k",
        type=float,
        nargs="+",
        default=[10.0, 30.0],
        help="Top-k heatmap percentages to occlude.",
    )
    parser.add_argument(
        "--random-seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Random occlusion seeds.",
    )
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument(
        "--output-json",
        default="artifacts/ttecam_occlusion/idrid/results.json",
    )
    parser.add_argument(
        "--summary-csv",
        default="artifacts/ttecam_occlusion/idrid/summary.csv",
    )
    return parser.parse_args()


def loader_for_dataset(dataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )


def main() -> None:
    args = parse_args()
    device = initialize(args)
    root = project_root()
    heatmap_dir = root / args.heatmap_dir
    if not heatmap_dir.is_dir():
        raise FileNotFoundError(
            f"TTECAM heatmap directory not found: {heatmap_dir}. "
            "Run ttecam_experiment/generate_ttecam_heatmaps.py first."
        )

    base_dataset = build_idrid_test_dataset(args)
    baseline_loader = build_loader(base_dataset, args)
    model = load_model(args, device)
    temperature = load_temperature(args)

    results = {
        "dataset": "IDRiD official test split",
        "method": "TTECAM",
        "heatmap_dir": str(heatmap_dir),
        "temperature": temperature,
        "mc_passes": args.mc_passes,
        "conditions": {},
    }
    summary_rows = []

    baseline = evaluate_condition_deterministic(
        model=model,
        loader=baseline_loader,
        device=device,
        temperature=temperature,
    )

    results["conditions"]["baseline"] = baseline
    baseline_true_probs = baseline["true_class_probs"]
    summary_rows.append({"condition": "baseline", **baseline})

    for top_k in args.top_k:
        targeted_dataset = HeatmapOccludedDataset(
            base_dataset=base_dataset,
            heatmap_dir=str(heatmap_dir),
            top_k_percent=top_k,
        )
        targeted = evaluate_condition_deterministic(
            model=model,
            loader=loader_for_dataset(targeted_dataset, args),
            device=device,
            temperature=temperature,
            baseline_true_probs=baseline_true_probs,
        )

        targeted_key = f"ttecam_top_{top_k:g}"
        results["conditions"][targeted_key] = targeted
        summary_rows.append(
            {
                "condition": "ttecam",
                "top_k_percent": top_k,
                **targeted,
            }
        )

        random_results = []
        for seed in args.random_seeds:
            random_dataset = RandomMeanOccludedDataset(
                base_dataset=base_dataset,
                top_k_percent=top_k,
                base_seed=seed,
            )
            random_result = evaluate_condition_deterministic(
                model=model,
                loader=loader_for_dataset(random_dataset, args),
                device=device,
                temperature=temperature,
                baseline_true_probs=baseline_true_probs,
            )
            
            random_result["seed"] = seed
            random_results.append(random_result)
            summary_rows.append(
                {
                    "condition": "random",
                    "top_k_percent": top_k,
                    "seed": seed,
                    **random_result,
                }
            )
        results["conditions"][f"random_top_{top_k:g}"] = random_results

    output_json = root / args.output_json
    summary_csv = root / args.summary_csv
    write_results_json(output_json, results)
    write_summary_csv(summary_csv, summary_rows)

    print(f"Saved full results to {output_json}")
    print(f"Saved summary table to {summary_csv}")


if __name__ == "__main__":
    main()
