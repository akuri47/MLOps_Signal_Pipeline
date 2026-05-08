"""
MLOps-style batch job: rolling-mean signal pipeline on OHLCV data.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MLOps signal pipeline")
    parser.add_argument("--input",    required=True, help="Path to input CSV file")
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path for log file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler (detailed)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Stream handler (INFO+ to stderr so stdout stays clean for metrics JSON)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    if not isinstance(cfg["seed"], int):
        raise TypeError(f"Config 'seed' must be an integer, got {type(cfg['seed'])}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"Config 'window' must be a positive integer, got {cfg['window']}")
    if not isinstance(cfg["version"], str) or not cfg["version"].strip():
        raise ValueError("Config 'version' must be a non-empty string")

    logger.info(
        "Config loaded — version=%s  seed=%d  window=%d",
        cfg["version"], cfg["seed"], cfg["window"],
    )
    return cfg


# ---------------------------------------------------------------------------
# Dataset loading & validation
# ---------------------------------------------------------------------------

def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError("Input file is empty or contains no parseable data")
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Input dataset has zero rows after parsing")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. "
            f"Columns present: {df.columns.tolist()}"
        )

    if df["close"].isnull().all():
        raise ValueError("Column 'close' contains only null values")

    logger.info("Dataset loaded — %d rows, %d columns", len(df), len(df.columns))
    logger.debug("Columns: %s", df.columns.tolist())
    return df


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def compute_rolling_mean(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.Series:
    """
    Compute rolling mean of 'close' with the given window.
    The first (window-1) rows produce NaN; they are excluded from signal
    computation downstream.
    """
    rolling_mean = df["close"].rolling(window=window, min_periods=window).mean()
    valid_count = rolling_mean.notna().sum()
    logger.info(
        "Rolling mean computed — window=%d  valid_rows=%d  NaN_rows=%d",
        window, valid_count, len(rolling_mean) - valid_count,
    )
    return rolling_mean


def generate_signal(close: pd.Series, rolling_mean: pd.Series, logger: logging.Logger) -> pd.Series:
    """
    signal = 1 where close > rolling_mean, else 0.
    Rows where rolling_mean is NaN are excluded (signal = NaN).
    """
    signal = pd.Series(np.nan, index=close.index)
    mask = rolling_mean.notna()
    signal[mask] = (close[mask] > rolling_mean[mask]).astype(int)
    logger.info(
        "Signal generated — rows_with_signal=%d  signal_1_count=%d  signal_0_count=%d",
        mask.sum(),
        int((signal[mask] == 1).sum()),
        int((signal[mask] == 0).sum()),
    )
    return signal


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_success_metrics(cfg: dict, signal: pd.Series, latency_ms: float) -> dict:
    valid_signal = signal.dropna()
    return {
        "version": cfg["version"],
        "rows_processed": int(valid_signal.count()),
        "metric": "signal_rate",
        "value": round(float(valid_signal.mean()), 4),
        "latency_ms": round(latency_ms),
        "seed": cfg["seed"],
        "status": "success",
    }


def build_error_metrics(cfg: dict | None, error_message: str) -> dict:
    base = {
        "status": "error",
        "error_message": error_message,
    }
    if cfg is not None:
        base["version"] = cfg.get("version", "unknown")
    else:
        base["version"] = "unknown"
    return base


def write_metrics(metrics: dict, output_path: str, logger: logging.Logger):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    logger = setup_logging(args.log_file)

    start_time = time.monotonic()
    logger.info("=== Job start ===")
    logger.info(
        "Args — input=%s  config=%s  output=%s  log=%s",
        args.input, args.config, args.output, args.log_file,
    )

    cfg = None
    try:
        # 1. Load & validate config
        cfg = load_config(args.config, logger)

        # 2. Set seed for reproducibility
        np.random.seed(cfg["seed"])
        logger.info("NumPy seed set to %d", cfg["seed"])

        # 3. Load & validate dataset
        df = load_dataset(args.input, logger)

        # 4. Compute rolling mean
        rolling_mean = compute_rolling_mean(df, cfg["window"], logger)

        # 5. Generate binary signal
        signal = generate_signal(df["close"], rolling_mean, logger)

        # 6. Compute metrics
        latency_ms = (time.monotonic() - start_time) * 1000
        metrics = build_success_metrics(cfg, signal, latency_ms)

        logger.info(
            "Metrics summary — rows_processed=%d  signal_rate=%.4f  latency_ms=%.0f",
            metrics["rows_processed"], metrics["value"], metrics["latency_ms"],
        )

        write_metrics(metrics, args.output, logger)
        logger.info("=== Job end — status=success ===")

        # Print final metrics to stdout (Docker requirement)
        print(json.dumps(metrics, indent=2))
        sys.exit(0)

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        error_metrics = build_error_metrics(cfg, str(exc))
        write_metrics(error_metrics, args.output, logger)
        logger.info("=== Job end — status=error ===")
        print(json.dumps(error_metrics, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
