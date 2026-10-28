import argparse
from pathlib import Path

import pandas as pd


METHODS = ["APE", "GDA_CLIP", "TIMO", "Tip_Adapter"]
DEFAULT_NOISE_LEVELS = [2, 4, 8, 16, 32]
KEY_COLUMNS = ["dataset", "shots", "seed", "method"]

# Change these paths if your logging code uses different folder names.
MODE_DIRS = {
    "base": "original",
    "tgi": "tgi",
    "igt": "igt",
    "both": "both",
}

NOISE_BASE_DIR = "original"


def parse_log_file(path, method):
    """Parse log lines in the format: dataset_shots_seed: score."""
    records = []

    with path.open("r") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line or ":" not in line:
                continue

            experiment, value = line.rsplit(":", 1)
            experiment = experiment.strip()

            try:
                score = float(value.strip())
            except ValueError:
                print(
                    f"Skipping non-numeric score at "
                    f"{path}:{line_number}: {line}"
                )
                continue

            parts = experiment.split("_")

            if len(parts) < 3:
                print(
                    f"Skipping invalid experiment key at "
                    f"{path}:{line_number}: {experiment}"
                )
                continue

            try:
                shots = int(parts[-2])
                seed = int(parts[-1])
            except ValueError:
                print(
                    f"Skipping invalid shot/seed at "
                    f"{path}:{line_number}: {experiment}"
                )
                continue

            dataset = "_".join(parts[:-2])

            records.append({
                "dataset": dataset,
                "shots": shots,
                "seed": seed,
                "method": method,
                "accuracy": score,
            })

    return records


def read_results(folder):
    """Read the classification-method logs from a results folder."""
    records = []

    for method in METHODS:
        path = folder / f"{method}.txt"

        if not path.exists():
            print(f"Warning: missing file {path}")
            continue

        records.extend(parse_log_file(path, method))

    df = pd.DataFrame(records)

    if df.empty:
        return df

    duplicates = df.duplicated(KEY_COLUMNS, keep=False)

    if duplicates.any():
        print(f"\nWARNING: duplicate experiment keys in {folder}:")
        print(
            df.loc[
                duplicates, KEY_COLUMNS + ["accuracy"]
            ].to_string(index=False)
        )
        raise ValueError(
            f"Duplicate results found in {folder}. "
            "Resolve duplicates before analysing."
        )

    return df


def save_csv(df, path):
    """Save a DataFrame and report its location."""
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ============================================================
# BASELINE AND ABLATION ANALYSIS
# ============================================================

def analyse_classification(root, mode):
    """Analyse baseline or TGI/IGT ablation results."""
    input_dir = root / MODE_DIRS[mode]
    output_dir = root / "analysis" / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {input_dir}"
        )

    df = read_results(input_dir)

    if df.empty:
        raise RuntimeError(
            f"No classification results found in {input_dir}"
        )

    # Convert the long-format records into one row per experiment.
    wide = df.pivot(
        index=["dataset", "shots", "seed"],
        columns="method",
        values="accuracy",
    ).reset_index()

    wide.columns.name = None

    for method in METHODS:
        if method not in wide.columns:
            wide[method] = float("nan")

    wide = wide[
        ["dataset", "shots", "seed"] + METHODS
    ]

    wide = wide.sort_values(
        ["dataset", "shots", "seed"]
    ).reset_index(drop=True)

    # 1. Detailed experiment results.
    save_csv(
        wide,
        output_dir / "all_results.csv",
    )

    # 2. Mean and standard deviation across seeds.
    summary = (
        wide.groupby(["dataset", "shots"])[METHODS]
        .agg(["mean", "std"])
        .round(3)
    )

    summary_path = output_dir / "summary_by_dataset_shots.csv"
    summary.to_csv(summary_path)
    print(f"Saved: {summary_path}")

    # 3. Best-performing method for each dataset and shot count.
    mean_scores = (
        wide.groupby(["dataset", "shots"])[METHODS]
        .mean()
    )

    comparison = mean_scores.reset_index()
    comparison["best_method"] = mean_scores.idxmax(axis=1).values
    comparison["best_score"] = mean_scores.max(axis=1).values

    save_csv(
        comparison,
        output_dir / "method_comparison.csv",
    )

    # 4. Overall method ranking.
    overall = wide[METHODS].agg(["mean", "std"]).T
    overall.columns = ["mean_accuracy", "std_accuracy"]

    overall = overall.sort_values(
        "mean_accuracy", ascending=False
    ).round(3)

    overall_path = output_dir / "overall_method_ranking.csv"
    overall.to_csv(overall_path)
    print(f"Saved: {overall_path}")

    print(f"\nOverall method ranking ({mode}):")
    print(overall.to_string())

    print(f"\nAnalysis complete. Results are in: {output_dir}")


# ============================================================
# JUNK-PROMPT ROBUSTNESS ANALYSIS
# ============================================================

def load_weight_logs(root, noise_levels):
    """
    Export junk-weight logs for inspection.

    The values are preserved without assuming their meaning.
    """
    records = []

    folders = [(NOISE_BASE_DIR, 0)] + [
        (f"original_junk{n}", n)
        for n in noise_levels
    ]

    for folder_name, noise in folders:
        folder = root / folder_name

        for filename in [
            "junk_equal_share.txt",
            "junk_weight_share.txt",
        ]:
            path = folder / filename

            if not path.exists():
                continue

            with path.open("r") as file:
                for line_number, line in enumerate(file, start=1):
                    line = line.strip()

                    if not line:
                        continue

                    entry = None
                    value = None

                    if ":" in line:
                        entry, raw_value = line.rsplit(":", 1)
                        entry = entry.strip()

                        try:
                            value = float(raw_value.strip())
                        except ValueError:
                            pass

                    records.append({
                        "folder": folder_name,
                        "noise": noise,
                        "file": filename,
                        "line_number": line_number,
                        "entry": entry if entry is not None else line,
                        "value": value,
                        "raw_line": line,
                    })

    return pd.DataFrame(records)


def analyse_junk(root, noise_levels):
    """Compare clean baseline results with junk-prompt results."""
    baseline_dir = root / NOISE_BASE_DIR
    output_dir = root / "analysis" / "junk_prompt"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_dir.exists():
        raise FileNotFoundError(
            f"Clean baseline directory not found: {baseline_dir}"
        )

    baseline = read_results(baseline_dir)

    if baseline.empty:
        raise RuntimeError(
            f"No baseline results found in {baseline_dir}"
        )

    baseline = baseline.rename(
        columns={"accuracy": "baseline_accuracy"}
    )

    all_runs = []

    for noise in noise_levels:
        folder = root / f"original_junk{noise}"

        if not folder.exists():
            print(f"Warning: missing noise folder {folder}")
            continue

        noisy = read_results(folder)

        if noisy.empty:
            print(f"Warning: no method results in {folder}")
            continue

        noisy = noisy.rename(
            columns={"accuracy": "noisy_accuracy"}
        )
        noisy["noise"] = noise

        merged = noisy.merge(
            baseline[
                KEY_COLUMNS + ["baseline_accuracy"]
            ],
            on=KEY_COLUMNS,
            how="left",
            validate="many_to_one",
            indicator=True,
        )

        unmatched = merged["_merge"] != "both"

        if unmatched.any():
            print(
                f"\nWarning: {unmatched.sum()} runs in {folder} "
                "have no matching clean baseline."
            )
            print(
                merged.loc[
                    unmatched, KEY_COLUMNS
                ].to_string(index=False)
            )

        merged = merged.drop(columns="_merge")

        # Accuracy drop is measured in percentage points if
        # the logged accuracy values are percentages.
        merged["accuracy_drop"] = (
            merged["baseline_accuracy"]
            - merged["noisy_accuracy"]
        )

        # Do not calculate relative degradation without a baseline.
        merged["relative_drop_pct"] = (
            100
            * merged["accuracy_drop"]
            / merged["baseline_accuracy"]
        )

        merged.loc[
            merged["baseline_accuracy"].isna()
            | (merged["baseline_accuracy"] == 0),
            "relative_drop_pct",
        ] = float("nan")

        all_runs.append(merged)

    if not all_runs:
        raise RuntimeError(
            "No noisy results were found. "
            "Check the junk-prompt folders and noise levels."
        )

    df = pd.concat(all_runs, ignore_index=True)

    df = df.sort_values(
        ["noise", "dataset", "shots", "seed", "method"]
    )

    # 1. Individual noisy runs matched to clean baselines.
    save_csv(
        df,
        output_dir / "noise_degradation.csv",
    )

    # 2. Overall robustness summary by noise level and method.
    summary = (
        df.groupby(["noise", "method"], dropna=False)
        .agg(
            total_runs=("noisy_accuracy", "count"),
            matched_runs=("baseline_accuracy", "count"),
            baseline_accuracy=("baseline_accuracy", "mean"),
            noisy_accuracy=("noisy_accuracy", "mean"),
            accuracy_std=("noisy_accuracy", "std"),
            mean_drop=("accuracy_drop", "mean"),
            drop_std=("accuracy_drop", "std"),
            relative_drop_pct=("relative_drop_pct", "mean"),
        )
        .reset_index()
        .round(3)
    )

    save_csv(
        summary,
        output_dir / "method_robustness_summary.csv",
    )

    # 3. Dataset-specific breakdown by noise and shot count.
    dataset_summary = (
        df.groupby(
            ["dataset", "shots", "noise", "method"],
            dropna=False,
        )
        .agg(
            total_runs=("noisy_accuracy", "count"),
            matched_runs=("baseline_accuracy", "count"),
            baseline_accuracy=("baseline_accuracy", "mean"),
            noisy_accuracy=("noisy_accuracy", "mean"),
            mean_drop=("accuracy_drop", "mean"),
            drop_std=("accuracy_drop", "std"),
            relative_drop_pct=("relative_drop_pct", "mean"),
        )
        .reset_index()
        .round(3)
    )

    save_csv(
        dataset_summary,
        output_dir / "degradation_by_dataset_shots.csv",
    )

    # 4. TIMO shot sensitivity, averaged across datasets.
    timo = df[df["method"] == "TIMO"].copy()

    if not timo.empty:
        timo_summary = (
            timo.groupby(["noise", "shots"], dropna=False)
            .agg(
                total_runs=("noisy_accuracy", "count"),
                matched_runs=("baseline_accuracy", "count"),
                baseline_accuracy=("baseline_accuracy", "mean"),
                noisy_accuracy=("noisy_accuracy", "mean"),
                mean_drop=("accuracy_drop", "mean"),
                drop_std=("accuracy_drop", "std"),
                relative_drop_pct=("relative_drop_pct", "mean"),
            )
            .reset_index()
            .round(3)
        )

        save_csv(
            timo_summary,
            output_dir / "timo_shot_sensitivity.csv",
        )

        print("\nTIMO mean accuracy drop by noise and shot count:")
        print(
            timo_summary.pivot(
                index="noise",
                columns="shots",
                values="mean_drop",
            ).to_string()
        )

        # Identify the shot setting with the largest mean drop
        # at each noise level, ignoring missing values.
        valid_timo = timo_summary.dropna(subset=["mean_drop"])

        if not valid_timo.empty:
            most_affected = (
                valid_timo.loc[
                    valid_timo.groupby("noise")["mean_drop"].idxmax()
                ][["noise", "shots", "mean_drop"]]
                .rename(columns={
                    "shots": "most_affected_shots",
                    "mean_drop": "largest_mean_drop",
                })
            )

            save_csv(
                most_affected,
                output_dir / "timo_most_affected_shots.csv",
            )

    # 5. Export junk-weight logs without assuming their semantics.
    weights = load_weight_logs(root, noise_levels)

    if not weights.empty:
        save_csv(
            weights,
            output_dir / "junk_weight_analysis.csv",
        )

    print("\nMean accuracy drop by method and noise level:")
    print(
        summary.pivot(
            index="noise",
            columns="method",
            values="mean_drop",
        ).to_string()
    )

    print(f"\nAnalysis complete. Results are in: {output_dir}")


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse baseline, TGI/IGT ablations, "
                    "or junk-prompt robustness experiments."
    )

    parser.add_argument(
        "--mode",
        choices=["base", "tgi", "igt", "both", "junk"],
        default="base",
        help="Experiment mode to analyse (default: base).",
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing experiment results.",
    )

    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=int,
        default=DEFAULT_NOISE_LEVELS,
        help="Junk-prompt counts to analyse (default: 2 4 8 16 32).",
    )

    args = parser.parse_args()

    if args.mode == "junk":
        if any(n < 0 for n in args.noise_levels):
            parser.error("Noise levels must be non-negative integers.")

        analyse_junk(args.root, args.noise_levels)
    else:
        analyse_classification(args.root, args.mode)