"""Agent Readiness Score (ARS) — Composite 0–100 metric for quantized model selection.

Reads the *_all_summary.csv files produced by the benchmark pipeline and
computes a single score per model-variant per agent, plus a cross-agent
aggregate.  Outputs:
  1. Per-agent ARS CSV   (e.g. ars_AGENT_1_SUPPORT.csv)
  2. Cross-agent ARS CSV (ars_overall.csv)
  3. Human-readable model card ranking (model_cards.txt)

Usage:
    python compute_ars.py --results_dir ./results/benchmark_20260730_143910
    python compute_ars.py --results_dir ./results/benchmark_20260730_143910 --target_vram 16
    python compute_ars.py --results_dir ./results/benchmark_20260730_143910 --preset edge
"""

import os
import re
import glob
import argparse
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Weight presets
# ─────────────────────────────────────────────────────────────────────────────
WEIGHT_PRESETS = {
    "default": {
        "R": 0.35, "P": 0.25, "E": 0.20, "D": 0.15, "C": 0.05,
        "description": "Safety-critical: prioritises task reliability above all",
    },
    "edge": {
        "R": 0.25, "P": 0.15, "E": 0.15, "D": 0.40, "C": 0.05,
        "description": "Edge / consumer GPU: prioritises low VRAM footprint",
    },
    "throughput": {
        "R": 0.20, "P": 0.15, "E": 0.35, "D": 0.15, "C": 0.15,
        "description": "High-throughput batch: prioritises speed and efficiency",
    },
    "cost": {
        "R": 0.25, "P": 0.15, "E": 0.20, "D": 0.10, "C": 0.30,
        "description": "Cost-optimised datacenter: prioritises power and GPU utilisation",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm(series: pd.Series, lower_is_better: bool = True) -> pd.Series:
    """Min-max normalise to [0, 1].  Returns 0.5 if range is zero."""
    series = pd.to_numeric(series, errors="coerce")
    mn, mx = series.min(), series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=series.index)
    normed = (series - mn) / (mx - mn)
    if lower_is_better:
        normed = 1.0 - normed
    return normed.fillna(0.0).clip(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Column-name resolution — handles the whitespace-padded CSV headers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and string values."""
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def _col(df: pd.DataFrame, *candidates: str) -> str:
    """Return the first column name from *candidates* that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def _num_col(df: pd.DataFrame, *candidates: str, fill: float | None = None) -> pd.Series:
    """Return a numeric (float) Series for the first matching candidate column.

    Coerces empty strings, whitespace, and non-numeric values to NaN,
    then optionally fills NaN values with `fill`.
    """
    col_name = _col(df, *candidates)
    series = pd.to_numeric(df[col_name], errors="coerce")
    if fill is not None:
        series = series.fillna(fill)
    return series


# ─────────────────────────────────────────────────────────────────────────────
# Sub-score computations
# ─────────────────────────────────────────────────────────────────────────────

def compute_reliability(df: pd.DataFrame) -> pd.Series:
    """Pillar 1 — Task Reliability (R).  Already 0–1."""
    success_rate = _num_col(df, "Success Rate", fill=0.0)
    shortcut_rate = _num_col(df, "Shortcut_Rate", fill=0.0)
    return (success_rate * (1.0 - shortcut_rate)).fillna(0.0).clip(0.0, 1.0)


def compute_precision(df: pd.DataFrame) -> pd.Series:
    """Pillar 2 — Tool Precision (P).  Already 0–1."""
    correct_order = _num_col(df, "Correct_Order_Rate", fill=0.0)
    efficiency = _num_col(df, "Tool_Call_Efficiency mean", "Tool_Call_Efficiency", fill=0.0).clip(0.0, 1.0)

    steps = _num_col(df, "Steps mean", "Steps", fill=1.0).replace(0, 1)
    parse_errors = _num_col(df, "Parse_Errors mean", "Parse_Errors", fill=0.0)
    parse_ok = (1.0 - (parse_errors / steps)).fillna(0.0).clip(0.0, 1.0)

    num_trials = _num_col(df, "Number of Trials", fill=0.0)
    perfect_count = _num_col(df, "Path_PERFECT_Count", fill=0.0)
    perfect_rate = (perfect_count / num_trials.replace(0, 1)).fillna(0.0).clip(0.0, 1.0)

    precision = (0.30 * correct_order
                 + 0.25 * efficiency
                 + 0.20 * parse_ok.where(num_trials > 0, 0.0)
                 + 0.25 * perfect_rate.where(num_trials > 0, 0.0)).fillna(0.0).clip(0.0, 1.0)
    return precision.where(num_trials > 0, 0.0)


def compute_efficiency(df: pd.DataFrame) -> pd.Series:
    """Pillar 3 — Efficiency (E).  Normalised across models."""
    tokens_step = _num_col(df, "Tokens_per_Useful_Step mean", "Tokens_per_Useful_Step")
    energy = _num_col(df, "Total Energy (J) mean", "Total Energy (J)")
    inf_time = _num_col(df, "Inference Time (s) mean", "Inference Time (s)")
    num_trials = _num_col(df, "Number of Trials", fill=0.0)

    eff = (0.40 * _norm(tokens_step, lower_is_better=True)
           + 0.30 * _norm(energy, lower_is_better=True)
           + 0.30 * _norm(inf_time, lower_is_better=True)).fillna(0.0).clip(0.0, 1.0)
    return eff.where(num_trials > 0, 0.0)


def compute_deployability(df: pd.DataFrame, target_vram: float | None = None) -> pd.Series:
    """Pillar 4 — Deployability (D).  Normalised across models."""
    peak_vram = _num_col(df, "Peak VRAM (GB) mean", "Peak VRAM (GB)")
    weights_gb = _num_col(df, "Weights Size (GB) mean", "Weights Size (GB)")

    if target_vram is not None:
        headroom = ((target_vram - peak_vram) / target_vram).fillna(0.0).clip(0.0, 1.0)
        # Models that exceed target VRAM get headroom = 0
    else:
        headroom = _norm(peak_vram, lower_is_better=True)

    return (0.50 * _norm(peak_vram, lower_is_better=True)
            + 0.30 * _norm(weights_gb, lower_is_better=True)
            + 0.20 * headroom).fillna(0.0).clip(0.0, 1.0)


def compute_operational_cost(df: pd.DataFrame) -> pd.Series:
    """Pillar 5 — Operational Cost (C).  Normalised across models."""
    avg_power = _num_col(df, "Avg Power (W) mean", "Avg Power (W)")
    avg_util = _num_col(df, "Avg GPU Util (%) mean", "Avg GPU Util (%)")
    num_trials = _num_col(df, "Number of Trials", fill=0.0)

    cost = (0.60 * _norm(avg_power, lower_is_better=True)
            + 0.40 * _norm(avg_util, lower_is_better=False)).fillna(0.0).clip(0.0, 1.0)
    return cost.where(num_trials > 0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# ARS computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ars(df: pd.DataFrame,
                weights: dict | None = None,
                target_vram: float | None = None) -> pd.DataFrame:
    """Compute ARS for every row (model variant) in a summary DataFrame.

    Returns a copy of df with added columns:
        R, P, E, D, C, ARS, ARS_Label, Best_For
    """
    if weights is None:
        weights = WEIGHT_PRESETS["default"]

    df = df.copy()

    # Ensure Params (B) is numeric
    if "Params (B)" in df.columns:
        df["Params (B)"] = _num_col(df, "Params (B)", fill=0.0)

    # Sub-scores
    df["R"] = compute_reliability(df)
    df["P"] = compute_precision(df)
    df["E"] = compute_efficiency(df)
    df["D"] = compute_deployability(df, target_vram)
    df["C"] = compute_operational_cost(df)

    # Composite
    df["ARS"] = (100.0 * (
        weights["R"] * df["R"]
        + weights["P"] * df["P"]
        + weights["E"] * df["E"]
        + weights["D"] * df["D"]
        + weights["C"] * df["C"]
    )).round(1)

    # Labels
    df["ARS_Label"] = df["ARS"].apply(_ars_label)
    df["Best_For"] = df.apply(_best_for_tag, axis=1)

    # VRAM incompatibility flag
    if target_vram is not None:
        peak_vram = _num_col(df, "Peak VRAM (GB) mean", "Peak VRAM (GB)")
        df["VRAM_Compatible"] = (peak_vram <= target_vram) & (peak_vram.notna())

    return df.sort_values("ARS", ascending=False)


def _ars_label(score: float) -> str:
    if score >= 85:
        return "🟢 Production Ready"
    elif score >= 70:
        return "🟡 Capable"
    elif score >= 50:
        return "🟠 Limited"
    elif score >= 30:
        return "🔴 Experimental"
    else:
        return "⛔ Not Recommended"


def _best_for_tag(row: pd.Series) -> str:
    r, p, e, d = row["R"], row["P"], row["E"], row["D"]
    if r > 0.9 and p > 0.8:
        return "Complex multi-step reasoning tasks"
    elif r > 0.7 and e > 0.8:
        return "High-throughput batch processing"
    elif d > 0.9 and r > 0.5:
        return "Edge deployment on consumer GPUs"
    elif e > 0.9:
        return "Low-latency single-step lookups"
    elif r > 0.5:
        return "Standard tool-calling workflows"
    elif d > 0.7:
        return "Lightweight experimentation"
    else:
        return "Experimental / testing only"


# ─────────────────────────────────────────────────────────────────────────────
# Cross-agent aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_overall_ars(per_agent_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average ARS and sub-scores across agents for each model variant.

    Args:
        per_agent_frames: {agent_id: ars_dataframe, ...}

    Returns:
        DataFrame indexed by (Model, Quantization, Params (B)) with averaged scores.
    """
    key_cols = ["Model", "Quantization", "Params (B)"]
    score_cols = ["R", "P", "E", "D", "C", "ARS"]

    parts = []
    for agent_id, adf in per_agent_frames.items():
        subset = adf[key_cols + score_cols].copy()
        subset["Agent"] = agent_id
        parts.append(subset)

    combined = pd.concat(parts, ignore_index=True)
    overall = combined.groupby(key_cols)[score_cols].mean().reset_index()
    overall["ARS"] = overall["ARS"].round(1)
    overall["ARS_Label"] = overall["ARS"].apply(_ars_label)
    overall["Best_For"] = overall.apply(_best_for_tag, axis=1)
    overall["Agents_Tested"] = len(per_agent_frames)

    # Also add per-agent ARS as separate columns
    for agent_id, adf in per_agent_frames.items():
        short_id = agent_id.replace("AGENT_", "").replace("_", " ").title()
        mapping = adf.set_index(key_cols)["ARS"]
        overall[f"ARS ({short_id})"] = overall.set_index(key_cols).index.map(
            lambda idx: mapping.get(idx, np.nan)
        )

    return overall.sort_values("ARS", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable model card
# ─────────────────────────────────────────────────────────────────────────────

def format_model_cards(df: pd.DataFrame, title: str = "Model Ranking") -> str:
    """Generate a human-readable model card ranking string."""
    lines = []
    lines.append(f"{'═' * 72}")
    lines.append(f"  {title}")
    lines.append(f"{'═' * 72}")

    for rank, (_, row) in enumerate(df.iterrows(), 1):
        model = row["Model"]
        quant = row["Quantization"]
        ars = row["ARS"]
        label = row["ARS_Label"]
        best_for = row["Best_For"]

        bar_len = int(ars / 100 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)

        peak_vram = None
        for col_name in ["Peak VRAM (GB) mean", "Peak VRAM (GB)"]:
            if col_name in row.index:
                val = pd.to_numeric(row[col_name], errors="coerce")
                if not pd.isna(val):
                    peak_vram = float(val)
                break

        tok_sec = None
        for col_name in ["Tokens / Sec mean", "Tokens / Sec"]:
            if col_name in row.index:
                val = pd.to_numeric(row[col_name], errors="coerce")
                if not pd.isna(val):
                    tok_sec = float(val)
                break

        lines.append("")
        lines.append(f"  #{rank}  {model}  [{quant}]")
        lines.append(f"       ARS: {ars:.1f}/100  {bar}  {label}")
        lines.append(f"       ✅ Best for: {best_for}")

        hw_parts = []
        if peak_vram is not None and not pd.isna(peak_vram):
            hw_parts.append(f"VRAM: {peak_vram:.1f} GB")
        if tok_sec is not None and not pd.isna(tok_sec):
            hw_parts.append(f"Speed: {tok_sec:.1f} tok/s")
        if hw_parts:
            lines.append(f"       ⚡ {'  |  '.join(hw_parts)}")

        # Sub-score breakdown
        sub_parts = []
        for col, lbl in [("R", "Reliability"), ("P", "Precision"),
                         ("E", "Efficiency"), ("D", "Deploy"), ("C", "Cost")]:
            if col in row.index and not pd.isna(row[col]):
                val = pd.to_numeric(row[col], errors="coerce")
                if not pd.isna(val):
                    sub_parts.append(f"{lbl}={float(val):.2f}")
        if sub_parts:
            lines.append(f"       📊 {' | '.join(sub_parts)}")

        # VRAM compatibility warning
        if "VRAM_Compatible" in row.index and not row.get("VRAM_Compatible", True):
            lines.append(f"       ⚠️  EXCEEDS TARGET VRAM — may not fit on device")

        lines.append(f"  {'─' * 68}")

    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# File discovery and main
# ─────────────────────────────────────────────────────────────────────────────

def discover_summary_csvs(results_dir: str) -> dict[str, str]:
    """Find *_all_summary.csv files grouped by agent_id.

    Returns:
        {agent_id: csv_path, ...}
    """
    pattern = os.path.join(results_dir, "**", "*_all_summary.csv")
    found = {}
    for path in glob.glob(pattern, recursive=True):
        # Extract agent_id from filename:  benchmark_AGENT_1_SUPPORT_all_summary.csv
        basename = os.path.basename(path)
        match = re.search(r"benchmark_(AGENT_\d+_\w+)_all_summary\.csv", basename)
        if match:
            found[match.group(1)] = path
    return found


def main():
    parser = argparse.ArgumentParser(
        description="Compute Agent Readiness Score (ARS) from benchmark results."
    )
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to the benchmark run directory (e.g. results/benchmark_20260730_143910)")
    parser.add_argument("--target_vram", type=float, default=None,
                        help="Target device VRAM in GB (e.g. 8, 16, 24). "
                             "Models exceeding this are flagged as incompatible.")
    parser.add_argument("--preset", type=str, default="default",
                        choices=list(WEIGHT_PRESETS.keys()),
                        help="Weight preset for the ARS computation.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (defaults to results_dir).")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)

    weights = WEIGHT_PRESETS[args.preset]
    print(f"ARS Preset: {args.preset} — {weights['description']}")
    print(f"Weights: R={weights['R']}, P={weights['P']}, E={weights['E']}, "
          f"D={weights['D']}, C={weights['C']}")
    if args.target_vram:
        print(f"Target VRAM: {args.target_vram} GB")
    print()

    # Discover CSVs
    csvs = discover_summary_csvs(results_dir)
    if not csvs:
        print(f"Error: No *_all_summary.csv files found in {results_dir}")
        return

    print(f"Found {len(csvs)} agent summary file(s):")
    for agent_id, path in sorted(csvs.items()):
        print(f"  {agent_id}: {path}")
    print()

    # Compute per-agent ARS
    per_agent_frames = {}
    for agent_id, csv_path in sorted(csvs.items()):
        print(f"{'─' * 72}")
        print(f"Computing ARS for: {agent_id}")
        print(f"{'─' * 72}")

        df = pd.read_csv(csv_path)
        df = _strip_cols(df)

        ars_df = compute_ars(df, weights=weights, target_vram=args.target_vram)
        per_agent_frames[agent_id] = ars_df

        # Save per-agent ARS CSV
        ars_csv_path = os.path.join(output_dir, f"ars_{agent_id}.csv")
        output_cols = ["Model", "Quantization", "Params (B)",
                       "R", "P", "E", "D", "C", "ARS", "ARS_Label", "Best_For"]
        if "VRAM_Compatible" in ars_df.columns:
            output_cols.append("VRAM_Compatible")
        ars_df[output_cols].to_csv(ars_csv_path, index=False)
        print(f"  Saved: {ars_csv_path}")

        # Print model cards
        cards = format_model_cards(ars_df, title=f"Agent Readiness Score — {agent_id}")
        print(cards)

        # Save model card text
        card_path = os.path.join(output_dir, f"ars_{agent_id}_cards.txt")
        with open(card_path, "w") as f:
            f.write(cards)

    # Cross-agent overall ARS
    if len(per_agent_frames) > 1:
        print(f"\n{'═' * 72}")
        print(f"  OVERALL ARS (averaged across {len(per_agent_frames)} agents)")
        print(f"{'═' * 72}\n")

        overall_df = compute_overall_ars(per_agent_frames)

        overall_csv = os.path.join(output_dir, "ars_overall.csv")
        overall_df.to_csv(overall_csv, index=False)
        print(f"Saved: {overall_csv}")

        overall_cards = format_model_cards(overall_df, title="Overall Agent Readiness Score")
        print(overall_cards)

        card_path = os.path.join(output_dir, "ars_overall_cards.txt")
        with open(card_path, "w") as f:
            f.write(overall_cards)
        print(f"Saved: {card_path}")

    print("\n✅ ARS computation complete.")


if __name__ == "__main__":
    main()
