#!/usr/bin/env python3
"""Compare test AUROC / balanced accuracy across linear-probe result trees.

Two local result trees (e.g. ``results_norelu`` vs ``results``) are compared
per task, and optionally against a challenge leaderboard summary CSV (one
aggregated row per dataset/disease/team).

Only ``test_metrics_val_loss.csv`` files are read from the local trees (the
val-loss checkpoint-selection strategy).  Layout expected under each root::

    <root>/<dataset>/<encoder>/results/<task>/fold<k>/test_metrics_val_loss.csv
    <root>/<dataset>/<encoder>/results/<task>/test_metrics_val_loss.csv   # no folds (MSWAL)

Folds are pooled per task (mean +/- sample std over the available folds); a
task with a single un-foldered CSV is reported as-is.  Runs still in progress
simply show as ``n/a``.

Examples::

    python src/analysis/compare_results.py --name-a norelu --name-b relu
    python src/analysis/compare_results.py --summary ~/Downloads/summary.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("auroc", "balanced_acc")
METRIC_LABELS = {"auroc": "AUROC", "balanced_acc": "BalancedAcc"}
CSV_NAME = "test_metrics_val_loss.csv"
REF_TEAM = "ctclip_lp"

# Local task directory names do not match the leaderboard's `disease` column.
# Everything is lowercased and the dataset-specific suffix dropped first; these
# are the residual renames that normalisation alone cannot recover.
DISEASE_ALIASES = {
    ("Gaozb_lung_part1", "emphysema_bulla"): "emphysema",
    ("MSWAL", "liver_tumor"): "liver_lesion",
}
TASK_SUFFIXES = ("_gaozblunghu",)


def normalize_task(dataset: str, task: str) -> str:
    """Map a local task directory name onto the leaderboard `disease` key."""
    key = task.lower()
    for suffix in TASK_SUFFIXES:
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    key = re.sub(r"_+", "_", key).strip("_")
    return DISEASE_ALIASES.get((dataset, key), key).lower()


def read_metric_csv(path: Path) -> dict[str, float] | None:
    """Read one test-metrics CSV, returning the wanted metrics of its last row."""
    try:
        frame = pd.read_csv(path)
    except Exception as error:  # noqa: BLE001 - surface the file, keep scanning
        print(f"[warn] failed to read {path}: {error}", file=sys.stderr)
        return None

    if frame.empty:
        print(f"[warn] empty metrics file: {path}", file=sys.stderr)
        return None

    missing = [metric for metric in METRICS if metric not in frame.columns]
    if missing:
        print(f"[warn] {path} lacks column(s) {missing}", file=sys.stderr)
        return None

    row = frame.iloc[-1]
    fold = row["fold"] if "fold" in frame.columns else np.nan
    return {
        **{metric: float(row[metric]) for metric in METRICS},
        "fold": float(fold) if pd.notna(fold) else np.nan,
    }


def fold_label(csv_path: Path, task_dir: Path) -> str:
    """Fold name from the directory between the task dir and the CSV ('-' if flat)."""
    relative = csv_path.parent.relative_to(task_dir)
    return "-" if relative == Path(".") else str(relative)


def collect_root(root: Path) -> pd.DataFrame:
    """Scan one result tree into a tidy per-fold frame."""
    if not root.is_dir():
        raise FileNotFoundError(f"result root does not exist: {root}")

    records: list[dict[str, object]] = []
    for csv_path in sorted(root.rglob(CSV_NAME)):
        parts = csv_path.relative_to(root).parts
        # <dataset>/<encoder>/results/<task>/[<fold>/]<csv>
        if len(parts) < 5 or parts[2] != "results":
            print(f"[warn] unexpected layout, skipping: {csv_path}", file=sys.stderr)
            continue

        dataset, encoder, task = parts[0], parts[1], parts[3]
        metrics = read_metric_csv(csv_path)
        if metrics is None:
            continue

        records.append(
            {
                "dataset": dataset,
                "encoder": encoder,
                "task": task,
                "disease": normalize_task(dataset, task),
                "fold": fold_label(csv_path, root / dataset / encoder / "results" / task),
                "fold_id": metrics["fold"],
                **{metric: metrics[metric] for metric in METRICS},
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=["dataset", "encoder", "task", "disease", "fold", "fold_id", *METRICS],
    )


def pool_folds(per_fold: pd.DataFrame, label: str) -> pd.DataFrame:
    """Mean / std / n over folds for every (dataset, encoder, task)."""
    columns = [
        "dataset",
        "encoder",
        "task",
        "disease",
        *[f"{metric}_{stat}_{label}" for metric in METRICS for stat in ("mean", "std")],
        f"n_folds_{label}",
    ]
    if per_fold.empty:
        return pd.DataFrame(columns=columns)

    grouped = per_fold.groupby(["dataset", "encoder", "task", "disease"], as_index=False).agg(
        **{
            f"{metric}_{stat}_{label}": (metric, stat)
            for metric in METRICS
            for stat in ("mean", "std")
        },
        **{f"n_folds_{label}": ("auroc", "size")},
    )
    # pandas std over a single fold is NaN; keep it NaN and let the formatter hide it.
    return grouped[columns]


def load_reference(path: Path, team: str, strategy: str | None) -> pd.DataFrame:
    """Load one team's aggregated leaderboard rows as reference metrics."""
    frame = pd.read_csv(path)
    required = {"dataset", "disease", "team", "strategy", "test_auroc", "test_balanced_acc"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks column(s) {missing}")

    selected = frame[frame["team"] == team]
    if strategy is not None:
        selected = selected[selected["strategy"] == strategy]
    if selected.empty:
        available = ", ".join(sorted(frame["team"].unique()))
        raise ValueError(f"no rows for team={team!r} (strategy={strategy!r}) in {path}; teams: {available}")

    duplicated = selected.duplicated(["dataset", "disease"], keep=False)
    if duplicated.any():
        dupes = selected.loc[duplicated, ["dataset", "disease", "strategy"]]
        print(
            f"[warn] {path} has multiple {team} rows per disease; keeping the last of:\n{dupes.to_string(index=False)}",
            file=sys.stderr,
        )

    reduced = selected.drop_duplicates(["dataset", "disease"], keep="last")
    # The summary is inconsistently cased across datasets (`Lung_Cancer` vs
    # `gallstone`), so join on the same lowercased key the local side uses.
    return pd.DataFrame(
        {
            "dataset": reduced["dataset"].to_numpy(),
            "disease": reduced["disease"].str.lower().to_numpy(),
            "ref_disease": reduced["disease"].to_numpy(),
            "auroc_mean_ref": reduced["test_auroc"].to_numpy(dtype=float),
            "balanced_acc_mean_ref": reduced["test_balanced_acc"].to_numpy(dtype=float),
            "ref_strategy": reduced["strategy"].to_numpy(),
        }
    )


def build_comparison(
    pooled_a: pd.DataFrame,
    pooled_b: pd.DataFrame,
    reference: pd.DataFrame | None,
) -> pd.DataFrame:
    """Outer-join the local trees, attach the reference, and add all deltas."""
    merged = pooled_a.merge(pooled_b, on=["dataset", "encoder", "task", "disease"], how="outer")

    if reference is None:
        joined = merged.assign(
            **{f"{metric}_mean_ref": np.nan for metric in METRICS},
            ref_strategy=pd.NA,
        )
    else:
        # Left join keeps only diseases seen locally, i.e. those in A and B together.
        joined = merged.merge(reference, on=["dataset", "disease"], how="left")

    deltas = {f"{metric}_delta_a_b": joined[f"{metric}_mean_a"] - joined[f"{metric}_mean_b"] for metric in METRICS}
    deltas |= {f"{metric}_delta_a_ref": joined[f"{metric}_mean_a"] - joined[f"{metric}_mean_ref"] for metric in METRICS}
    deltas |= {f"{metric}_delta_b_ref": joined[f"{metric}_mean_b"] - joined[f"{metric}_mean_ref"] for metric in METRICS}

    return joined.assign(**deltas).sort_values(["dataset", "task"]).reset_index(drop=True)


def macro_average(comparison: pd.DataFrame, sides: tuple[str, ...]) -> pd.DataFrame:
    """Macro-average per dataset over tasks where every requested side is present."""
    needed = [f"{metric}_mean_{side}" for metric in METRICS for side in sides]
    both = comparison.dropna(subset=needed)
    if both.empty:
        return both

    return both.groupby("dataset", as_index=False).agg(
        n_tasks=("task", "size"),
        **{
            f"{metric}_mean_{side}": (f"{metric}_mean_{side}", "mean")
            for metric in METRICS
            for side in sides
        },
    )


def format_value(mean: float, std: float, n_folds: float) -> str:
    """``mean±std [#folds]``; a single run has no spread, so print the mean alone."""
    if pd.isna(mean):
        return "n/a"
    if pd.isna(n_folds):  # reference row: already pooled upstream
        return f"{mean:.4f}"
    folds = int(n_folds)
    if folds <= 1 or pd.isna(std):
        return f"{mean:.4f}       [1]"
    return f"{mean:.4f}±{std:.4f} [{folds}]"


def format_delta(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:+.4f}"


def print_metric_table(
    comparison: pd.DataFrame,
    metric: str,
    names: dict[str, str],
    with_reference: bool,
) -> None:
    columns: list[tuple[str, int, callable]] = [
        ("dataset", 18, lambda row: str(row["dataset"])),
        ("task", 46, lambda row: str(row["task"])),
        (names["a"], 21, lambda row: format_value(row[f"{metric}_mean_a"], row[f"{metric}_std_a"], row["n_folds_a"])),
        (names["b"], 21, lambda row: format_value(row[f"{metric}_mean_b"], row[f"{metric}_std_b"], row["n_folds_b"])),
        (f"Δ({names['a']}-{names['b']})", 16, lambda row: format_delta(row[f"{metric}_delta_a_b"])),
    ]
    if with_reference:
        columns += [
            (names["ref"], 10, lambda row: format_value(row[f"{metric}_mean_ref"], np.nan, np.nan)),
            (f"Δ({names['a']}-ref)", 16, lambda row: format_delta(row[f"{metric}_delta_a_ref"])),
            (f"Δ({names['b']}-ref)", 16, lambda row: format_delta(row[f"{metric}_delta_b_ref"])),
        ]

    header = " ".join(f"{title:<{width}}" for title, width, _ in columns)
    print(f"\n=== test {METRIC_LABELS[metric]} " + "=" * max(0, len(header) - len(METRIC_LABELS[metric]) - 10))
    print(header)
    print("-" * len(header))
    for _, row in comparison.iterrows():
        print(" ".join(f"{extract(row):<{width}}" for _, width, extract in columns))


def print_summary(summary: pd.DataFrame, sides: tuple[str, ...], names: dict[str, str], caption: str) -> None:
    if summary.empty:
        print(f"\n{caption}: no dataset has tasks present in all of these - nothing to average.")
        return

    print(f"\n{caption}")
    columns = ["dataset", "#tasks"] + [
        f"{METRIC_LABELS[metric]} {names[side]}" for metric in METRICS for side in sides
    ]
    widths = [18, 7] + [max(14, len(title) + 2) for title in columns[2:]]
    header = " ".join(f"{title:<{width}}" for title, width in zip(columns, widths))
    print(header)
    print("-" * len(header))
    for _, row in summary.iterrows():
        cells = [str(row["dataset"]), str(int(row["n_tasks"]))] + [
            f"{row[f'{metric}_mean_{side}']:.4f}" for metric in METRICS for side in sides
        ]
        print(" ".join(f"{cell:<{width}}" for cell, width in zip(cells, widths)))


def report_coverage(comparison: pd.DataFrame, names: dict[str, str], with_reference: bool) -> None:
    for side, other in (("a", "b"), ("b", "a")):
        only = comparison[comparison[f"auroc_mean_{side}"].notna() & comparison[f"auroc_mean_{other}"].isna()]
        if not only.empty:
            tasks = ", ".join(f"{row.dataset}/{row.task}" for row in only.itertuples())
            print(f"\n[note] only in {names[side]}: {tasks}")

    if with_reference:
        unmatched = comparison[comparison["auroc_mean_ref"].isna()]
        if not unmatched.empty:
            tasks = ", ".join(f"{row.dataset}/{row.task} (key={row.disease})" for row in unmatched.itertuples())
            print(f"\n[note] no {names['ref']} reference row for: {tasks}")

    mismatched = comparison[
        comparison["auroc_mean_a"].notna()
        & comparison["auroc_mean_b"].notna()
        & (comparison["n_folds_a"] != comparison["n_folds_b"])
    ]
    for row in mismatched.itertuples():
        print(
            f"[note] fold-count mismatch {row.dataset}/{row.task}: "
            f"{names['a']}={int(row.n_folds_a)} vs {names['b']}={int(row.n_folds_b)}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--a", type=Path, default=Path("results_norelu"), help="first result root (default: results_norelu)")
    parser.add_argument("--b", type=Path, default=Path("results"), help="second result root (default: results)")
    parser.add_argument("--name-a", default=None, help="label for --a (default: its directory name)")
    parser.add_argument("--name-b", default=None, help="label for --b (default: its directory name)")
    parser.add_argument("--summary", type=Path, default=None, help="leaderboard summary CSV to compare against")
    parser.add_argument("--summary-team", default=REF_TEAM, help=f"team column value to use as reference (default: {REF_TEAM})")
    parser.add_argument(
        "--summary-strategy",
        default=None,
        help="restrict reference rows to this strategy (default: any; the summary uses one per dataset)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the per-task comparison to this CSV")
    parser.add_argument("--per-fold-out", type=Path, default=None, help="write the raw per-fold rows to this CSV")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = {
        "a": args.name_a or args.a.name,
        "b": args.name_b or args.b.name,
        "ref": args.summary_team,
    }

    try:
        per_fold_a = collect_root(args.a)
        per_fold_b = collect_root(args.b)
        reference = load_reference(args.summary, args.summary_team, args.summary_strategy) if args.summary else None
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if per_fold_a.empty and per_fold_b.empty:
        print(f"error: no {CSV_NAME} found under {args.a} or {args.b}", file=sys.stderr)
        return 1

    comparison = build_comparison(pool_folds(per_fold_a, "a"), pool_folds(per_fold_b, "b"), reference)
    with_reference = reference is not None

    print(f"A = {args.a}  ({names['a']})")
    print(f"B = {args.b}  ({names['b']})")
    if with_reference:
        print(f"ref = {args.summary}  (team={args.summary_team}, already pooled by the leaderboard)")
    print(f"local metric source: {CSV_NAME}; cells are mean±std [#folds], std is sample std (ddof=1)")

    for metric in METRICS:
        print_metric_table(comparison, metric, names, with_reference)

    print_summary(
        macro_average(comparison, ("a", "b")),
        ("a", "b"),
        names,
        f"Macro-average over tasks present in both {names['a']} and {names['b']}:",
    )
    if with_reference:
        print_summary(
            macro_average(comparison, ("a", "ref")),
            ("a", "ref"),
            names,
            f"Macro-average over tasks present in both {names['a']} and {names['ref']}:",
        )
        print_summary(
            macro_average(comparison, ("b", "ref")),
            ("b", "ref"),
            names,
            f"Macro-average over tasks present in both {names['b']} and {names['ref']}:",
        )

    report_coverage(comparison, names, with_reference)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.out, index=False)
        print(f"\nwrote per-task comparison -> {args.out}")

    if args.per_fold_out is not None:
        args.per_fold_out.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(
            [per_fold_a.assign(root=names["a"]), per_fold_b.assign(root=names["b"])],
            ignore_index=True,
        )
        combined.to_csv(args.per_fold_out, index=False)
        print(f"wrote per-fold rows      -> {args.per_fold_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
