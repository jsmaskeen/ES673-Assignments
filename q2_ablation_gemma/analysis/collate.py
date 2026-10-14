import argparse
import glob
import json
import os
from typing import List

import pandas as pd


def load_all_records(input_dir: str) -> pd.DataFrame:
    """Load all Parquet and JSON files from input_dir into a single DataFrame."""
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    dfs: List[pd.DataFrame] = []

    if parquet_files:
        print(f"[collate] Found {len(parquet_files)} Parquet files in '{input_dir}'")
        for f in parquet_files:
            try:
                df = pd.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                print(f"[collate] Warning: Could not read {f}: {e}")
    else:
        # Fallback to JSON
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        print(f"[collate] Found {len(json_files)} JSON files in '{input_dir}'")
        for f in json_files:
            try:
                with open(f, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                    if isinstance(data, list):
                        dfs.append(pd.DataFrame(data))
                    elif isinstance(data, dict):
                        dfs.append(pd.DataFrame([data]))
            except Exception as e:
                print(f"[collate] Warning: Could not read {f}: {e}")

    if not dfs:
        raise ValueError(f"No valid Parquet or JSON results found in '{input_dir}'.")

    combined = pd.concat(dfs, ignore_index=True)
    # Deduplicate runs with identical parameters if re-run
    key_cols = [c for c in ["task", "surgery_type", "perturbation_type", "severity", "head_mode", "component", "position", "seed"] if c in combined.columns]
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    return combined


def aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Group over experimental configurations and compute mean and std across seeds."""
    group_cols = [c for c in ["task", "surgery_type", "perturbation_type", "head_mode", "component", "position", "severity"] if c in df.columns]
    
    numeric_metric_cols = [
        c for c in df.columns
        if c.startswith("metric_") and pd.api.types.is_numeric_dtype(df[c])
    ]
    non_numeric_metric_cols = [
        c for c in df.columns
        if c.startswith("metric_") and not pd.api.types.is_numeric_dtype(df[c])
    ]

    # Calculate seed counts per configuration
    counts = df.groupby(group_cols, as_index=False).size().rename(columns={"size": "num_seeds"})

    agg_dict = {}
    for m in numeric_metric_cols:
        agg_dict[m] = ["mean", "std"]
    for m in non_numeric_metric_cols:
        agg_dict[m] = "first"

    aggregated = df.groupby(group_cols, as_index=False).agg(agg_dict)
    
    # Flatten multi-level columns
    flat_cols = []
    for col in aggregated.columns:
        if isinstance(col, tuple):
            if col[1] and col[1] not in ("", "first"):
                flat_cols.append(f"{col[0]}_{col[1]}")
            else:
                flat_cols.append(col[0])
        else:
            flat_cols.append(col)
    aggregated.columns = flat_cols

    # Merge num_seeds
    aggregated = pd.merge(counts, aggregated, on=group_cols, how="right")

    # Replace all NaN values in std columns with 0.0 (since variation across 1 seed is 0.0)
    std_cols = [c for c in aggregated.columns if c.endswith("_std")]
    aggregated[std_cols] = aggregated[std_cols].fillna(0.0)

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Collate and aggregate Gemma-2 ablation results")
    parser.add_argument("--input-dir", type=str, default="outputs", help="Directory containing run outputs")
    parser.add_argument("--output-csv", type=str, default="outputs/summary_all_runs.csv", help="Path for raw collated CSV")
    parser.add_argument("--output-agg", type=str, default="outputs/summary_aggregated.csv", help="Path for aggregated CSV")
    parser.add_argument("--output-md", type=str, default="outputs/summary_table.md", help="Path for markdown summary table")
    args = parser.parse_args()

    df_raw = load_all_records(args.input_dir)
    df_raw.to_csv(args.output_csv, index=False)
    print(f"[collate] Saved raw collated records ({len(df_raw)} rows) to: {args.output_csv}")

    df_agg = aggregate_runs(df_raw)
    df_agg.to_csv(args.output_agg, index=False)
    print(f"[collate] Saved aggregated records ({len(df_agg)} rows) to: {args.output_agg}")

    # Generate clean Markdown summary table with rounded floats and zeroed NaNs
    df_md = df_agg.copy()
    for col in df_md.select_dtypes(include=["float", "float64"]).columns:
        df_md[col] = df_md[col].round(4)
    df_md = df_md.fillna(0.0)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# Gemma-2 Ablation Study: Aggregated Summary\n\n")
        try:
            f.write(df_md.to_markdown(index=False))
        except Exception:
            # Fallback pure-python table writer
            header = " | ".join(str(c) for c in df_md.columns)
            sep = " | ".join(["---"] * len(df_md.columns))
            rows = [" | ".join(str(val) for val in row) for row in df_md.values]
            f.write(f"| {header} |\n| {sep} |\n" + "\n".join(f"| {r} |" for r in rows) + "\n")
    print(f"[collate] Saved markdown summary table to: {args.output_md}")


if __name__ == "__main__":
    main()
