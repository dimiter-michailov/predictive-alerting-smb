from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

SCOREBOARD_COLUMNS = [
    "family",
    "machine_id",
    "model_name",
    "window_size",
    "horizon",
    "threshold",
    "pr_auc",
    "brier_score",
    "false_alert_rate",
    "precision_at_threshold",
    "recall_at_threshold",
    "f1_at_threshold",
    "metadata_file",
]

SCOREBOARD_MULTI_COLUMNS = [
    "family",
    "machine_ids",
    "n_machines",
    "model_name",
    "window_size",
    "horizon",
    "threshold",
    "pr_auc",
    "brier_score",
    "false_alert_rate",
    "precision_at_threshold",
    "recall_at_threshold",
    "f1_at_threshold",
    "metadata_file",
]

def _get_metadata_dir():
    metadata_dir = Path("scoreboard_metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir

def _get_next_metadata_number():
    metadata_dir = _get_metadata_dir()
    existing_files = sorted(metadata_dir.glob("scoreboard_metadata_*.txt"))

    if not existing_files:
        return 1

    max_number = 0
    for file_path in existing_files:
        stem = file_path.stem
        try:
            number = int(stem.split("_")[-1])
            max_number = max(max_number, number)
        except ValueError:
            continue

    return max_number + 1

def write_metadata_file(metadata_text):
    metadata_dir = _get_metadata_dir()
    file_number = _get_next_metadata_number()
    metadata_file = metadata_dir / f"scoreboard_metadata_{file_number:03d}.txt"
    metadata_file.write_text(metadata_text, encoding="utf-8")
    return str(metadata_file.as_posix())

def log_classifier_holdout(machine_id, model_name, window_size, horizon, threshold,
                           holdout_metrics, metadata_text="", csv_path="scoreboard.csv"):
    metadata_file = ""
    if metadata_text:
        metadata_file = write_metadata_file(metadata_text)

    row = {
        "family": "classifier",
        "machine_id": machine_id,
        "model_name": model_name,
        "window_size": window_size,
        "horizon": horizon,
        "threshold": threshold,
        "pr_auc": holdout_metrics["pr_auc"],
        "brier_score": holdout_metrics["brier_score"],
        "false_alert_rate": holdout_metrics["false_alert_rate"],
        "precision_at_threshold": holdout_metrics["precision_at_threshold"],
        "recall_at_threshold": holdout_metrics["recall_at_threshold"],
        "f1_at_threshold": holdout_metrics["f1_at_threshold"],
        "metadata_file": metadata_file,
    }

    csv_file = Path(csv_path)

    if csv_file.exists() and csv_file.stat().st_size > 0:
        try:
            df = pd.read_csv(csv_file)
        except EmptyDataError:
            df = pd.DataFrame(columns=SCOREBOARD_COLUMNS)
    else:
        df = pd.DataFrame(columns=SCOREBOARD_COLUMNS)

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_file, index=False)

    return metadata_file

def log_classifier_multi_holdout(machine_ids, model_name, window_size, horizon, threshold,
                                 holdout_metrics, metadata_text="",
                                 csv_path="scoreboard_multi_summary.csv"):
    metadata_file = ""
    if metadata_text:
        metadata_file = write_metadata_file(metadata_text)

    row = {
        "family": "classifier_multi",
        "machine_ids": "|".join(machine_ids),
        "n_machines": len(machine_ids),
        "model_name": model_name,
        "window_size": window_size,
        "horizon": horizon,
        "threshold": threshold,
        "pr_auc": holdout_metrics["pr_auc"],
        "brier_score": holdout_metrics["brier_score"],
        "false_alert_rate": holdout_metrics["false_alert_rate"],
        "precision_at_threshold": holdout_metrics["precision_at_threshold"],
        "recall_at_threshold": holdout_metrics["recall_at_threshold"],
        "f1_at_threshold": holdout_metrics["f1_at_threshold"],
        "metadata_file": metadata_file,
    }

    csv_file = Path(csv_path)

    if csv_file.exists() and csv_file.stat().st_size > 0:
        try:
            df = pd.read_csv(csv_file)
        except EmptyDataError:
            df = pd.DataFrame(columns=SCOREBOARD_MULTI_COLUMNS)
    else:
        df = pd.DataFrame(columns=SCOREBOARD_MULTI_COLUMNS)

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv_file, index=False)

    return metadata_file