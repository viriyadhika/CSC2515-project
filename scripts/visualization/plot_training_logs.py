import ast
import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class RunMetrics:
    model: str
    pretrain_steps: int
    finetune_steps: int
    data_tag: str  # e.g. "full_set", "1percent"

    # x-axes
    train_epochs: List[float] = field(default_factory=list)
    eval_epochs: List[float] = field(default_factory=list)

    # y-axes
    train_loss: List[float] = field(default_factory=list)
    eval_loss: List[float] = field(default_factory=list)
    eval_accuracy: List[float] = field(default_factory=list)
    eval_balanced_accuracy: List[float] = field(default_factory=list)


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_log_file(path: str) -> Optional[RunMetrics]:
    """
    Parse one training log file of the form:
    <model>_<pretrain_steps>_<finetune_steps>_<full_set|1percent>.log
    and extract loss / eval metrics.
    """
    basename = os.path.basename(path)
    name, _ext = os.path.splitext(basename)
    parts = name.split("_")
    if len(parts) < 4:
        # Unexpected filename format
        return None

    model = parts[0]
    try:
        pretrain_steps = int(parts[1])
        finetune_steps = int(parts[2])
    except ValueError:
        # If steps cannot be parsed as ints, skip this file
        return None

    data_tag = "_".join(parts[3:])  # just in case there are extra underscores

    metrics = RunMetrics(
        model=model,
        pretrain_steps=pretrain_steps,
        finetune_steps=finetune_steps,
        data_tag=data_tag,
    )

    # We'll keep independent counters for train and eval sequences,
    # but prefer "epoch" if present in the log line.
    train_step_idx = 0
    eval_step_idx = 0

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "{" not in line or "}" not in line:
                continue

            # Extract the dictionary literal from the line
            dict_str = line[line.find("{") : line.rfind("}") + 1]
            try:
                record = ast.literal_eval(dict_str)
            except (SyntaxError, ValueError):
                continue

            if not isinstance(record, dict):
                continue

            epoch_val = _safe_float(record.get("epoch"))

            # Training loss lines have "loss" but usually no "eval_loss"
            if "loss" in record and "eval_loss" not in record:
                loss_val = _safe_float(record.get("loss"))
                if loss_val is not None:
                    x = epoch_val if epoch_val is not None else float(train_step_idx)
                    metrics.train_epochs.append(x)
                    metrics.train_loss.append(loss_val)
                    train_step_idx += 1

            # Evaluation lines have "eval_loss" and possibly eval metrics
            if "eval_loss" in record:
                eval_loss_val = _safe_float(record.get("eval_loss"))
                acc_val = _safe_float(record.get("eval_accuracy"))
                bal_acc_val = _safe_float(record.get("eval_balanced_accuracy"))

                x = epoch_val if epoch_val is not None else float(eval_step_idx)
                metrics.eval_epochs.append(x)

                if eval_loss_val is not None:
                    metrics.eval_loss.append(eval_loss_val)
                else:
                    metrics.eval_loss.append(float("nan"))

                if acc_val is not None:
                    metrics.eval_accuracy.append(acc_val)

                if bal_acc_val is not None:
                    metrics.eval_balanced_accuracy.append(bal_acc_val)

                eval_step_idx += 1

    # If we didn't parse anything meaningful, drop this run
    if not metrics.train_loss and not metrics.eval_loss:
        return None

    return metrics


def load_all_runs(data_dir: str) -> List[RunMetrics]:
    """
    Load all dino_/mae_/mae_freq_ logs from the given data directory.

    Returns a flat list of runs (one per log file).
    """
    pattern = os.path.join(data_dir, "*.log")
    paths = sorted(glob.glob(pattern))

    runs: List[RunMetrics] = []

    for path in paths:
        basename = os.path.basename(path)
        if not (
            basename.startswith("dino_")
            or basename.startswith("mae_")
            or basename.startswith("mae_freq_")
        ):
            continue

        metrics = parse_log_file(path)
        if metrics is None:
            continue
        runs.append(metrics)

    return runs


def plot_comparisons(runs: List[RunMetrics], output_dir: str) -> None:
    """
    For each data_tag (full_set, 1percent, ...), create a 2x2 grid:
    - eval_accuracy vs epoch for each model
    - eval_balanced_accuracy vs epoch for each model
    - train_loss vs epoch for each model
    - eval_loss vs epoch for each model
    """
    # Group runs by data_tag
    by_data: Dict[str, List[RunMetrics]] = {}
    for run in runs:
        by_data.setdefault(run.data_tag, []).append(run)

    os.makedirs(output_dir, exist_ok=True)

    for data_tag, group in by_data.items():
        if not group:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        (ax_acc, ax_bal_acc), (ax_train_loss, ax_eval_loss) = axes

        for run in group:
            label = f"{run.model} (pre={run.pretrain_steps}, ft={run.finetune_steps})"

            # Eval accuracy
            if run.eval_accuracy:
                ax_acc.plot(
                    run.eval_epochs[: len(run.eval_accuracy)],
                    run.eval_accuracy,
                    marker="o",
                    label=label,
                )

            # Eval balanced accuracy
            if run.eval_balanced_accuracy:
                ax_bal_acc.plot(
                    run.eval_epochs[: len(run.eval_balanced_accuracy)],
                    run.eval_balanced_accuracy,
                    marker="o",
                    label=label,
                )

            # Train loss
            if run.train_loss:
                ax_train_loss.plot(
                    run.train_epochs,
                    run.train_loss,
                    label=label,
                )

            # Eval loss
            if run.eval_loss:
                ax_eval_loss.plot(
                    run.eval_epochs[: len(run.eval_loss)],
                    run.eval_loss,
                    marker="o",
                    label=label,
                )

        # Titles and labels
        fig.suptitle(f"Training / Evaluation Metrics – {data_tag}", fontsize=14)

        ax_acc.set_title("Eval accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

        ax_bal_acc.set_title("Eval balanced accuracy")
        ax_bal_acc.set_xlabel("Epoch")
        ax_bal_acc.set_ylabel("Balanced accuracy")
        ax_bal_acc.legend()
        ax_bal_acc.grid(True, alpha=0.3)

        ax_train_loss.set_title("Training loss")
        ax_train_loss.set_xlabel("Epoch")
        ax_train_loss.set_ylabel("Loss")
        ax_train_loss.legend()
        ax_train_loss.grid(True, alpha=0.3)

        ax_eval_loss.set_title("Eval loss")
        ax_eval_loss.set_xlabel("Epoch")
        ax_eval_loss.set_ylabel("Loss")
        ax_eval_loss.legend()
        ax_eval_loss.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        outfile = os.path.join(output_dir, f"metrics_{data_tag}.png")
        fig.savefig(outfile, dpi=150)
        # Also show interactively if run from a notebook/terminal
        # Comment this out if you prefer only saving.
        plt.show()


def plot_pretrain_vs_scratch(runs: List[RunMetrics], output_dir: str) -> None:
    """
    For each (data_tag, model, finetune_steps), if we have both
    pretrain_steps == 0 and pretrain_steps == 50, create a comparison
    figure (2x2 grid) showing 0_xx vs 50_xx.
    """
    # Index runs by full configuration
    index: Dict[Tuple[str, str, int, int], RunMetrics] = {}
    for run in runs:
        key = (run.data_tag, run.model, run.finetune_steps, run.pretrain_steps)
        index[key] = run

    os.makedirs(output_dir, exist_ok=True)

    # Only care about fine-tune steps 40 and 80 as requested
    finetune_targets = {40, 80}

    # Build a set of (data_tag, model, ft) triples to inspect
    combos: set[Tuple[str, str, int]] = set()
    for (data_tag, model, ft, pre) in index.keys():
        if ft in finetune_targets:
            combos.add((data_tag, model, ft))

    for data_tag, model, ft in sorted(combos):
        run_0 = index.get((data_tag, model, ft, 0))
        run_50 = index.get((data_tag, model, ft, 50))

        if run_0 is None or run_50 is None:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        (ax_acc, ax_bal_acc), (ax_train_loss, ax_eval_loss) = axes

        max_epoch = 20

        for run, label in [(run_0, "pretrain=0"), (run_50, "pretrain=50")]:
            # Eval accuracy
            if run.eval_accuracy:
                acc_pairs = [
                    (e, a)
                    for e, a in zip(run.eval_epochs, run.eval_accuracy)
                    if e <= max_epoch
                ]
                if acc_pairs:
                    xs, ys = zip(*acc_pairs)
                    ax_acc.plot(xs, ys, marker="o", label=label)

            # Eval balanced accuracy
            if run.eval_balanced_accuracy:
                bal_pairs = [
                    (e, b)
                    for e, b in zip(run.eval_epochs, run.eval_balanced_accuracy)
                    if e <= max_epoch
                ]
                if bal_pairs:
                    xs, ys = zip(*bal_pairs)
                    ax_bal_acc.plot(xs, ys, marker="o", label=label)

            # Train loss
            if run.train_loss:
                train_pairs = [
                    (e, l)
                    for e, l in zip(run.train_epochs, run.train_loss)
                    if e <= max_epoch
                ]
                if train_pairs:
                    xs, ys = zip(*train_pairs)
                    ax_train_loss.plot(xs, ys, label=label)

            # Eval loss
            if run.eval_loss:
                loss_pairs = [
                    (e, l)
                    for e, l in zip(run.eval_epochs, run.eval_loss)
                    if e <= max_epoch
                ]
                if loss_pairs:
                    xs, ys = zip(*loss_pairs)
                    ax_eval_loss.plot(xs, ys, marker="o", label=label)

        title = f"{model} – {data_tag} – ft={ft}: pretrain 0 vs 50"
        fig.suptitle(title, fontsize=14)

        ax_acc.set_title("Eval accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

        ax_bal_acc.set_title("Eval balanced accuracy")
        ax_bal_acc.set_xlabel("Epoch")
        ax_bal_acc.set_ylabel("Balanced accuracy")
        ax_bal_acc.legend()
        ax_bal_acc.grid(True, alpha=0.3)

        ax_train_loss.set_title("Training loss")
        ax_train_loss.set_xlabel("Epoch")
        ax_train_loss.set_ylabel("Loss")
        ax_train_loss.legend()
        ax_train_loss.grid(True, alpha=0.3)

        ax_eval_loss.set_title("Eval loss")
        ax_eval_loss.set_xlabel("Epoch")
        ax_eval_loss.set_ylabel("Loss")
        ax_eval_loss.legend()
        ax_eval_loss.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        outfile = os.path.join(
            output_dir, f"pretrain_compare_{model}_{data_tag}_ft{ft}.png"
        )
        fig.savefig(outfile, dpi=150)
        plt.show()


def plot_final_balanced_accuracy_bars(runs: List[RunMetrics], output_dir: str) -> None:
    """
    Bar chart of final eval_balanced_accuracy for each (model, pretrain_steps),
    split by data_tag and finetune_steps so comparisons remain meaningful.
    """
    # Group by (data_tag, finetune_steps)
    grouped: Dict[Tuple[str, int], List[RunMetrics]] = {}
    for run in runs:
        key = (run.data_tag, run.finetune_steps)
        grouped.setdefault(key, []).append(run)

    os.makedirs(output_dir, exist_ok=True)

    for (data_tag, ft_steps), group in sorted(grouped.items()):
        labels: List[str] = []
        values: List[float] = []

        for run in group:
            if not run.eval_balanced_accuracy:
                continue
            final_bal_acc = run.eval_balanced_accuracy[-1]
            labels.append(f"{run.model}\npre={run.pretrain_steps}")
            values.append(final_bal_acc)

        if not labels:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        x = list(range(len(labels)))
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Final eval balanced accuracy")
        ax.set_title(f"Final balanced accuracy – {data_tag}, ft={ft_steps}")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        outfile = os.path.join(
            output_dir, f"final_bal_acc_{data_tag}_ft{ft_steps}.png"
        )
        fig.savefig(outfile, dpi=150)
        plt.show()


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "scripts", "visualization", "figures")

    runs = load_all_runs(data_dir)
    if not runs:
        print(f"No matching logs found under {data_dir}")
        return

    plot_comparisons(runs, output_dir)
    plot_pretrain_vs_scratch(runs, output_dir)
    plot_final_balanced_accuracy_bars(runs, output_dir)


if __name__ == "__main__":
    main()

