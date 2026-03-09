"""
Inference runtime benchmark + empirical slope estimation + throughput plots + training-time-only benchmark

This script runs THREE benchmarks:

(A) Inference throughput vs. test-set size N (batch inference on X_test[:N])
    - Measures elapsed time for predicting N samples at once.
    - Reports mean/median runtime and throughput (samples/s).
    - Fits empirical scaling exponent alpha from: log T = alpha log N + c (on the tail of N).
    - Produces two log-log plots:
        (1) total runtime (median_ms) vs N
        (2) throughput (median samples/s) vs N

(B) Inference throughput vs. TRAINING SET SIZE (n_train) at FIXED inference batch size N_fixed
    - Keeps inference batch size fixed and varies n_train.
    - For each n_train:
        * generate synthetic data with that training size
        * fit the model
        * measure inference throughput on a fixed test batch
    - Produces log-log plot: throughput (samples/s) vs n_train
    - Saves CSV: *_speed_vs_train_size.csv and PNG: *_speed_vs_train_size.png

(C) Training time ONLY vs. training set size (n_train)
    - No inference timing.
    - For each n_train:
        * generate synthetic data
        * fit the model
        * record fit_time_s
    - Produces log-log plot: fit_time_s vs n_train
    - Saves CSV: *_train_time_vs_train_size.csv and PNG: *_train_time_vs_train_size.png

Notes:
- Uses CUDA Events when available; otherwise perf_counter with synchronize.
- Special-cases MOH: fit_mixture_hypernetworks, predict_2

Run:
  python benchmark_time_vs_N.py --n-train 5000 --n-test 50000 --repeats 10 --warmup 5 --out results.csv

Outputs:
  - CSV (A): args.out (includes alpha, R^2, bootstrap CI)
  - PNG (A): *_scaling_total.png and *_throughput_vs_n.png
  - CSV (B): *_speed_vs_train_size.csv
  - PNG (B): *_speed_vs_train_size.png
  - CSV (C): *_train_time_vs_train_size.csv
  - PNG (C): *_train_time_vs_train_size.png
"""

import gc
import time
import json
import argparse
from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# -------------------------
# Repro + perf knobs
# -------------------------
def set_global_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def maybe_set_torch_perf_flags(deterministic: bool = False) -> None:
    if not TORCH_AVAILABLE:
        return
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def is_cuda_available() -> bool:
    return TORCH_AVAILABLE and torch.cuda.is_available()


def hard_sync_if_gpu() -> None:
    if is_cuda_available():
        torch.cuda.synchronize()


# -------------------------
# Timing
# -------------------------
class Timer:
    """
    Uses CUDA Events when available; falls back to perf_counter with sync.
    """
    def __init__(self, use_cuda_events: bool = True):
        self.use_cuda_events = use_cuda_events and is_cuda_available()

    def time_fn(self, fn: Callable[[], Any]) -> float:
        """Return elapsed seconds for fn()."""
        if self.use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            hard_sync_if_gpu()
            start.record()
            fn()
            end.record()
            hard_sync_if_gpu()
            return start.elapsed_time(end) / 1000.0
        else:
            hard_sync_if_gpu()
            t0 = time.perf_counter()
            fn()
            hard_sync_if_gpu()
            t1 = time.perf_counter()
            return t1 - t0


def detect_device(model: Any) -> str:
    if TORCH_AVAILABLE:
        try:
            if hasattr(model, "parameters"):
                p = next(model.parameters())
                return str(p.device)
        except Exception:
            pass
        for attr in ("device", "_device"):
            if hasattr(model, attr):
                try:
                    return str(getattr(model, attr))
                except Exception:
                    pass
    return "cpu"


# -------------------------
# Data
# -------------------------
def make_synthetic_tabular(
    n_train: int,
    n_test: int,
    n_features: int = 64,
    n_informative: int = 2,
    n_redundant: int = 2,
    n_classes: int = 2,
    class_sep: float = 1.0,
    weights: Optional[List[float]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=n_train + n_test,
        n_features=n_features,
        n_informative=min(n_informative, n_features),
        n_redundant=min(n_redundant, max(0, n_features - min(n_informative, n_features))),
        n_classes=n_classes,
        class_sep=class_sep,
        weights=weights,
        random_state=seed,
    )
    X = X.astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


# -------------------------
# Model adapters
# -------------------------
def get_fit_fn(model_name: str, model: Any) -> Callable[[np.ndarray, np.ndarray], Any]:
    if model_name.upper() == "MOH":
        if not hasattr(model, "fit_mixture_hypernetworks"):
            raise AttributeError("MOH model missing fit_mixture_hypernetworks(X, y).")
        return lambda X, y: model.fit_mixture_hypernetworks(X, y)
    if hasattr(model, "fit"):
        return lambda X, y: model.fit(X, y)
    raise AttributeError(f"{model_name} has no recognized fit method.")


def get_predict_fn(model_name: str, model: Any) -> Callable[[np.ndarray], Any]:
    if model_name.upper() == "MOH":
        if not hasattr(model, "predict_2"):
            raise AttributeError("MOH model missing predict_2(X).")
        return lambda X: model.predict_2(X)
    if hasattr(model, "predict"):
        return lambda X: model.predict(X)
    if hasattr(model, "predict_proba"):
        return lambda X: model.predict_proba(X)
    raise AttributeError(f"{model_name} has no recognized predict method.")


def maybe_set_eval_mode(model: Any) -> None:
    if TORCH_AVAILABLE and hasattr(model, "eval"):
        try:
            model.eval()
        except Exception:
            pass


# -------------------------
# Empirical slope estimation (log T = alpha log N + c)
# -------------------------
def _safe_log(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0):
        raise ValueError("Log requires all values > 0.")
    return np.log(x)


def fit_empirical_slope(
    n: np.ndarray,
    t_seconds: np.ndarray,
    fit_tail_fraction: float = 0.6,
) -> Dict[str, float]:
    n = np.asarray(n, dtype=np.float64)
    t_seconds = np.asarray(t_seconds, dtype=np.float64)

    idx = np.argsort(n)
    n = n[idx]
    t_seconds = t_seconds[idx]

    k = max(3, int(np.ceil(len(n) * fit_tail_fraction)))
    n_fit = n[-k:]
    t_fit = t_seconds[-k:]

    x = _safe_log(n_fit)
    y = _safe_log(t_fit)

    alpha, c = np.polyfit(x, y, 1)

    y_hat = alpha * x + c
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "alpha": float(alpha),
        "c": float(c),
        "r2": float(r2),
        "n_points_used": float(k),
        "n_min_fit": float(n_fit.min()),
        "n_max_fit": float(n_fit.max()),
    }


def bootstrap_alpha_ci(
    n: np.ndarray,
    t_seconds: np.ndarray,
    fit_tail_fraction: float = 0.6,
    n_boot: int = 500,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)

    n = np.asarray(n, dtype=np.float64)
    t_seconds = np.asarray(t_seconds, dtype=np.float64)

    idx = np.argsort(n)
    n = n[idx]
    t_seconds = t_seconds[idx]

    k = max(3, int(np.ceil(len(n) * fit_tail_fraction)))
    n_fit = n[-k:]
    t_fit = t_seconds[-k:]

    x = _safe_log(n_fit)
    y = _safe_log(t_fit)

    alphas = []
    for _ in range(n_boot):
        samp = rng.integers(0, k, size=k)
        xs = x[samp]
        ys = y[samp]
        if np.all(xs == xs[0]):
            continue
        a, _c = np.polyfit(xs, ys, 1)
        alphas.append(a)

    if len(alphas) < 10:
        return (float("nan"), float("nan"))

    alphas = np.sort(np.asarray(alphas, dtype=np.float64))
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    lo = float(np.quantile(alphas, lo_q))
    hi = float(np.quantile(alphas, hi_q))
    return lo, hi


def add_empirical_slopes_to_df(
    df: pd.DataFrame,
    time_col_ms: str = "median_ms",
    fit_tail_fraction: float = 0.6,
    add_bootstrap_ci: bool = True,
    bootstrap_n: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    out = df.copy()

    slope_rows = []
    for model, g in out.groupby("model"):
        g2 = g.sort_values("n_pred")
        n = g2["n_pred"].to_numpy(dtype=np.float64)
        t_seconds = (g2[time_col_ms].to_numpy(dtype=np.float64)) / 1000.0

        fit = fit_empirical_slope(n, t_seconds, fit_tail_fraction=fit_tail_fraction)

        if add_bootstrap_ci:
            lo, hi = bootstrap_alpha_ci(
                n, t_seconds,
                fit_tail_fraction=fit_tail_fraction,
                n_boot=bootstrap_n,
                ci=0.95,
                seed=seed,
            )
        else:
            lo, hi = (float("nan"), float("nan"))

        slope_rows.append({
            "model": model,
            "alpha": fit["alpha"],
            "alpha_ci_low": lo,
            "alpha_ci_high": hi,
            "alpha_r2": fit["r2"],
            "alpha_n_points_used": int(fit["n_points_used"]),
            "alpha_fit_n_min": int(fit["n_min_fit"]),
            "alpha_fit_n_max": int(fit["n_max_fit"]),
        })

    slopes = pd.DataFrame(slope_rows)
    out = out.merge(slopes, on="model", how="left")
    return out


# -------------------------
# Plotting helpers
# -------------------------
def _format_log_axes_plain(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="both")


def plot_scaling_total_and_throughput(
    df: pd.DataFrame,
    fit_tail_fraction: float,
    base_out: str,
    subtitle: str,
) -> None:
    """
    Two plots:
      1) total runtime (median_ms) vs N
      2) throughput (median samples/s) vs N
    """
    models = list(df["model"].unique())

    # ---------- Plot 1: total runtime ----------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    lines = ["Empirical slope (log T vs log N):"]

    for model in models:
        g = df[df["model"] == model].sort_values("n_pred")
        n = g["n_pred"].to_numpy(dtype=np.float64)
        t_ms = g["median_ms"].to_numpy(dtype=np.float64)

        ax.plot(n, t_ms, marker="o", linewidth=1.8, markersize=4, label=model)

        t_s = t_ms / 1000.0
        fit = fit_empirical_slope(n, t_s, fit_tail_fraction=fit_tail_fraction)
        alpha, c = fit["alpha"], fit["c"]

        n_line = np.linspace(n.min(), n.max(), 200)
        t_line_s = np.exp(alpha * np.log(n_line) + c)
        t_line_ms = 1000.0 * t_line_s
        ax.plot(n_line, t_line_ms, linestyle="--", linewidth=1.4)

        lo = g["alpha_ci_low"].iloc[0] if "alpha_ci_low" in g.columns else np.nan
        hi = g["alpha_ci_high"].iloc[0] if "alpha_ci_high" in g.columns else np.nan
        if pd.notna(lo) and pd.notna(hi):
            lines.append(f"{model}: α={alpha:.2f} [{float(lo):.2f}, {float(hi):.2f}], R²={fit['r2']:.2f}")
        else:
            lines.append(f"{model}: α={alpha:.2f}, R²={fit['r2']:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input size (N)")
    ax.set_ylabel("Runtime (median) [ms]")
    ax.set_title("Inference runtime vs input size (total)")
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    ax.grid(True, which="both", linewidth=0.6, alpha=0.35)
    _format_log_axes_plain(ax)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

    ax.text(
        0.02, 0.02, "\n".join(lines),
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", linewidth=0.8, alpha=0.85),
    )

    fig.tight_layout()
    out_total = f"{base_out}_scaling_total.png"
    fig.savefig(out_total, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {out_total}")
    plt.show()

    # ---------- Plot 2: throughput ----------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for model in models:
        g = df[df["model"] == model].sort_values("n_pred")
        n = g["n_pred"].to_numpy(dtype=np.float64)
        thr = g["throughput_median_samples_per_s"].to_numpy(dtype=np.float64)

        ax.plot(n, thr, marker="o", linewidth=1.8, markersize=4, label=model)

        # implied throughput curve from fitted total runtime: throughput = N / T(N)
        t_ms = g["median_ms"].to_numpy(dtype=np.float64)
        t_s = t_ms / 1000.0
        fit = fit_empirical_slope(n, t_s, fit_tail_fraction=fit_tail_fraction)
        alpha, c = fit["alpha"], fit["c"]
        n_line = np.linspace(n.min(), n.max(), 200)
        t_line_s = np.exp(alpha * np.log(n_line) + c)
        thr_line = n_line / t_line_s
        ax.plot(n_line, thr_line, linestyle="--", linewidth=1.4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input size (N)")
    ax.set_ylabel("Throughput (median) [samples/s]")
    ax.set_title("Inference throughput vs input size")
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    ax.grid(True, which="both", linewidth=0.6, alpha=0.35)
    _format_log_axes_plain(ax)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

    ax.text(
        0.02, 0.02, "\n".join(lines),
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", linewidth=0.8, alpha=0.85),
    )

    fig.tight_layout()
    out_thr = f"{base_out}_throughput_vs_n.png"
    fig.savefig(out_thr, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {out_thr}")
    plt.show()


def plot_speed_vs_train_size(
    df_train: pd.DataFrame,
    base_out: str,
    fixed_n_pred: int,
    subtitle: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    models = list(df_train["model"].unique())

    for model in models:
        g = df_train[df_train["model"] == model].sort_values("n_train")
        x = g["n_train"].to_numpy(dtype=np.float64)
        y = g["throughput_median_samples_per_s"].to_numpy(dtype=np.float64)
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=4, label=model)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training set size (n_train)")
    ax.set_ylabel("Inference throughput (median) [samples/s]")
    ax.set_title(f"Inference throughput vs training size (fixed N={fixed_n_pred})")
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    ax.grid(True, which="both", linewidth=0.6, alpha=0.35)
    _format_log_axes_plain(ax)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

    fig.tight_layout()
    outp = f"{base_out}_speed_vs_train_size.png"
    fig.savefig(outp, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {outp}")
    plt.show()


def plot_train_time_vs_train_size(
    df_fit: pd.DataFrame,
    base_out: str,
    subtitle: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    models = list(df_fit["model"].unique())

    for model in models:
        g = df_fit[df_fit["model"] == model].sort_values("n_train")
        x = g["n_train"].to_numpy(dtype=np.float64)
        y = g["fit_time_s"].to_numpy(dtype=np.float64)
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=4, label=model)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training set size (n_train)")
    ax.set_ylabel("Training time [s]")
    ax.set_title("Training time vs training size (fit only)")
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    ax.grid(True, which="both", linewidth=0.6, alpha=0.35)
    _format_log_axes_plain(ax)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

    fig.tight_layout()
    outp = f"{base_out}_train_time_vs_train_size.png"
    fig.savefig(outp, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {outp}")
    plt.show()


# -------------------------
# Benchmark core: inference time vs N
# -------------------------
@dataclass
class Row:
    model: str
    device: str
    n_pred: int
    mean_ms: float
    std_ms: float
    median_ms: float
    throughput_mean_samples_per_s: float
    throughput_median_samples_per_s: float
    fit_time_s: float


def benchmark_time_vs_n(
    model_name: str,
    model: Any,
    X_test: np.ndarray,
    n_values: List[int],
    repeats: int,
    warmup: int,
    timer: "Timer",
) -> List[Row]:
    predict_fn = get_predict_fn(model_name, model)
    device = detect_device(model)

    warm_n = n_values[min(len(n_values) - 1, 3)]
    for _ in range(warmup):
        _ = predict_fn(X_test[:warm_n])
    hard_sync_if_gpu()

    rows: List[Row] = []

    for n in n_values:
        Xn = X_test[:n]

        times = []
        for _ in range(repeats):
            gc.collect()
            if is_cuda_available():
                torch.cuda.empty_cache()
            times.append(timer.time_fn(lambda: predict_fn(Xn)))

        ts = np.array(times, dtype=np.float64)
        mean_s = float(ts.mean())
        median_s = float(np.median(ts))

        mean_ms = mean_s * 1000.0
        std_ms = float(ts.std(ddof=1) * 1000.0) if repeats > 1 else 0.0
        median_ms = median_s * 1000.0

        throughput_mean = float(n / mean_s) if mean_s > 0 else float("nan")
        throughput_median = float(n / median_s) if median_s > 0 else float("nan")

        rows.append(
            Row(
                model=model_name,
                device=device,
                n_pred=n,
                mean_ms=float(mean_ms),
                std_ms=float(std_ms),
                median_ms=float(median_ms),
                throughput_mean_samples_per_s=throughput_mean,
                throughput_median_samples_per_s=throughput_median,
                fit_time_s=float("nan"),
            )
        )

    return rows


# -------------------------
# Benchmark B: throughput vs training size at fixed N_pred
# -------------------------
def benchmark_speed_vs_train_size(
    model_name: str,
    model_factory: Callable[[], Any],
    n_train_values: List[int],
    fixed_n_pred: int,
    n_test: int,
    n_features: int,
    repeats: int,
    warmup: int,
    timer: Timer,
    seed: int,
) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []

    for ntr in n_train_values:
        # Ensure enough test points for fixed inference batch
        n_test_eff = max(n_test, fixed_n_pred)

        X_train, X_test, y_train, y_test = make_synthetic_tabular(
            n_train=int(ntr),
            n_test=int(n_test_eff),
            n_features=int(n_features),
            seed=seed,
            class_sep=1.0,
        )

        model = model_factory()
        fit_fn = get_fit_fn(model_name, model)
        predict_fn = get_predict_fn(model_name, model)

        t0 = time.perf_counter()
        fit_fn(X_train, y_train)
        hard_sync_if_gpu()
        t1 = time.perf_counter()
        fit_time_s = t1 - t0

        maybe_set_eval_mode(model)
        device = detect_device(model)

        Xn = X_test[:fixed_n_pred]

        # warmup
        for _ in range(warmup):
            _ = predict_fn(Xn)
        hard_sync_if_gpu()

        times = []
        for _ in range(repeats):
            gc.collect()
            if is_cuda_available():
                torch.cuda.empty_cache()
            times.append(timer.time_fn(lambda: predict_fn(Xn)))

        ts = np.array(times, dtype=np.float64)
        mean_s = float(ts.mean())
        median_s = float(np.median(ts))

        throughput_mean = float(fixed_n_pred / mean_s) if mean_s > 0 else float("nan")
        throughput_median = float(fixed_n_pred / median_s) if median_s > 0 else float("nan")

        out_rows.append({
            "model": model_name,
            "device": device,
            "n_train": int(ntr),
            "n_pred_fixed": int(fixed_n_pred),
            "n_features": int(n_features),
            "mean_ms": float(mean_s * 1000.0),
            "median_ms": float(median_s * 1000.0),
            "std_ms": float(ts.std(ddof=1) * 1000.0) if repeats > 1 else 0.0,
            "throughput_mean_samples_per_s": throughput_mean,
            "throughput_median_samples_per_s": throughput_median,
            "fit_time_s": float(fit_time_s),
        })

        del model
        gc.collect()
        if is_cuda_available():
            torch.cuda.empty_cache()

    return out_rows


# -------------------------
# Benchmark C: training time only vs training size
# -------------------------
def benchmark_train_time_vs_train_size(
    model_name: str,
    model_factory: Callable[[], Any],
    n_train_values: List[int],
    n_test: int,
    n_features: int,
    seed: int,
) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []

    for ntr in n_train_values:
        X_train, X_test, y_train, y_test = make_synthetic_tabular(
            n_train=int(ntr),
            n_test=int(n_test),
            n_features=int(n_features),
            seed=seed,
            class_sep=1.0,
        )

        model = model_factory()
        fit_fn = get_fit_fn(model_name, model)
        device = detect_device(model)

        t0 = time.perf_counter()
        fit_fn(X_train, y_train)
        hard_sync_if_gpu()
        t1 = time.perf_counter()
        fit_time_s = t1 - t0

        out_rows.append({
            "model": model_name,
            "device": device,
            "n_train": int(ntr),
            "n_test": int(n_test),
            "n_features": int(n_features),
            "fit_time_s": float(fit_time_s),
        })

        del model
        gc.collect()
        if is_cuda_available():
            torch.cuda.empty_cache()

    return out_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=25000)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--use-cuda-events", action="store_true", default=True)
    parser.add_argument("--no-cuda-events", action="store_false", dest="use_cuda_events")
    parser.add_argument("--out", type=str, default="runtime_time_vs_n.csv")

    # Slope/plot knobs
    parser.add_argument("--fit-tail-fraction", type=float, default=0.6)
    parser.add_argument("--bootstrap-n", type=int, default=500)

    # Benchmark B knobs
    parser.add_argument("--fixed-n-pred", type=int, default=5_000,
                        help="Fixed inference batch size for speed-vs-train-size test.")
    parser.add_argument("--train-grid", type=str, default="1000,2000,3000,4000,5000,6000,7000,8000,9000,10000",
                        help="Comma-separated list of n_train for speed-vs-train-size + train-time-only tests.")

    args = parser.parse_args()

    set_global_seeds(args.seed)
    maybe_set_torch_perf_flags(deterministic=False)

    # --------- EDIT THESE IMPORTS FOR YOUR PROJECT ----------
    from tabpfn import TabPFNClassifier
    from ticl.prediction import MotherNetClassifier, tabflex
    from tabpfn.constants import ModelVersion
    from xgboost import XGBClassifier
    # --------------------------------------------------------

    # --------- EDIT THESE MODELS FOR YOUR PROJECT ----------
    icl_model_factories: List[Tuple[str, Callable[[], Any]]] = [
        ("MOH", lambda: MoH_reversed_tree_based_big_boy_vfinal(
            m=6, overlap=0.15, verbose=False,
            minority_cluster='KMeans', majority_cluster='BalancedKMeansLSA'
        )),
        ("EnsembleMixtutre", lambda: CustomEnsembleClassifier(
            n_estimators=10,
            base_model_class=MoH_reversed_tree_based_big_boy_vfinal,
            model_params={
                "m": 6, "overlap": 0.15, "verbose": False, "random_state": 10,
                "minority_cluster": 'KMeans', "majority_cluster": 'BalancedKMeansLSA'
            },
            mode="regular"
        )),
        ("TabPFN", lambda: TabPFNClassifier.create_default_for_version(ModelVersion.V2)),
        ("MotherNet", lambda: MotherNetClassifier(device="cuda",inference_device="cuda")),
        ("TabFlex", lambda: tabflex.TabFlex()),
        ("XGBoost", lambda: XGBClassifier(device="cuda"))

    ]
    icl_models = [(name, factory()) for name, factory in icl_model_factories]
    # -------------------------------------------------------

    timer = Timer(use_cuda_events=args.use_cuda_events)

    # =========================
    # Benchmark A: inference time/throughput vs N
    # =========================
    X_train, X_test, y_train, y_test = make_synthetic_tabular(
        n_train=args.n_train,
        n_test=args.n_test,
        n_features=args.n_features,
        seed=args.seed,
        class_sep=1.0,
    )

    n_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    n_values = [n for n in n_values if n <= X_test.shape[0]]
    if X_test.shape[0] not in n_values:
        n_values.append(X_test.shape[0])
    n_values = sorted(set(n_values))

    meta = {
        "benchmark_A": {
            "n_train": args.n_train,
            "n_test": args.n_test,
            "n_features": args.n_features,
            "seed": args.seed,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "cuda_available": is_cuda_available(),
            "torch_version": torch.__version__ if TORCH_AVAILABLE else None,
            "n_values": n_values,
            "fit_tail_fraction": args.fit_tail_fraction,
            "bootstrap_n": args.bootstrap_n,
        },
        "benchmark_B": {
            "fixed_n_pred": args.fixed_n_pred,
            "train_grid": args.train_grid,
        },
        "benchmark_C": {
            "train_grid": args.train_grid,
        }
    }
    print("Benchmark config:", json.dumps(meta, indent=2))

    all_rows: List[Dict[str, Any]] = []

    for model_name, model in icl_models:
        print(f"\n=== Fitting: {model_name} ===")
        fit_fn = get_fit_fn(model_name, model)

        t0 = time.perf_counter()
        fit_fn(X_train, y_train)
        hard_sync_if_gpu()
        t1 = time.perf_counter()
        fit_time_s = t1 - t0

        maybe_set_eval_mode(model)

        print(f"Fit time: {fit_time_s:.3f} s | Device: {detect_device(model)}")
        print(f"--- Inference time vs N: {model_name} ---")

        rows = benchmark_time_vs_n(
            model_name=model_name,
            model=model,
            X_test=X_test,
            n_values=n_values,
            repeats=args.repeats,
            warmup=args.warmup,
            timer=timer,
        )

        for r in rows:
            r.fit_time_s = fit_time_s
            all_rows.append({
                "model": r.model,
                "device": r.device,
                "n_pred": r.n_pred,
                "mean_ms": r.mean_ms,
                "std_ms": r.std_ms,
                "median_ms": r.median_ms,
                "throughput_mean_samples_per_s": r.throughput_mean_samples_per_s,
                "throughput_median_samples_per_s": r.throughput_median_samples_per_s,
                "fit_time_s": r.fit_time_s,
            })
            print(
                f"N={r.n_pred:>6} | median={r.median_ms:>9.2f} ms | "
                f"throughput(med)={r.throughput_median_samples_per_s:>10.1f} samples/s"
            )

        del model
        gc.collect()
        if is_cuda_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)

    add_ci = args.bootstrap_n > 0
    df = add_empirical_slopes_to_df(
        df,
        time_col_ms="median_ms",
        fit_tail_fraction=args.fit_tail_fraction,
        add_bootstrap_ci=add_ci,
        bootstrap_n=max(args.bootstrap_n, 0),
        seed=args.seed,
    )

    df.to_csv(args.out, index=False)
    print(f"\nSaved benchmark A results to: {args.out}")

    base = args.out.replace(".csv", "")
    subtitle = f"{args.repeats} repeats, {args.warmup} warmup | tail={args.fit_tail_fraction:.2f}"
    plot_scaling_total_and_throughput(
        df=df,
        fit_tail_fraction=args.fit_tail_fraction,
        base_out=base,
        subtitle=subtitle,
    )

    # =========================
    # Benchmark B: throughput vs training size at fixed inference batch
    # =========================
    try:
        train_grid = [int(x.strip()) for x in args.train_grid.split(",") if x.strip()]
    except Exception as e:
        raise ValueError(f"Failed parsing --train-grid '{args.train_grid}': {e}")

    print("\n=== Benchmark B: throughput vs training size at fixed inference batch ===")
    print(f"Fixed N_pred: {args.fixed_n_pred}")
    print(f"n_train grid: {train_grid}")

    feat_rows_all: List[Dict[str, Any]] = []
    for model_name, factory in icl_model_factories:
        print(f"\n--- {model_name} ---")
        rows_b = benchmark_speed_vs_train_size(
            model_name=model_name,
            model_factory=factory,
            n_train_values=train_grid,
            fixed_n_pred=args.fixed_n_pred,
            n_test=args.n_test,
            n_features=args.n_features,
            repeats=args.repeats,
            warmup=args.warmup,
            timer=timer,
            seed=args.seed,
        )
        feat_rows_all.extend(rows_b)

        for r in rows_b:
            print(
                f"n_train={r['n_train']:>6} | "
                f"median={r['median_ms']:>9.2f} ms | "
                f"throughput(med)={r['throughput_median_samples_per_s']:>10.1f} samples/s | "
                f"fit={r['fit_time_s']:.2f}s"
            )

    df_train = pd.DataFrame(feat_rows_all)
    out_train_csv = f"{base}_speed_vs_train_size_v2.csv"
    df_train.to_csv(out_train_csv, index=False)
    print(f"\nSaved benchmark B results to: {out_train_csv}")

    subtitle_b = f"{args.repeats} repeats, {args.warmup} warmup | fixed N={args.fixed_n_pred} | n_features={args.n_features}"
    plot_speed_vs_train_size(
        df_train=df_train,
        base_out=base,
        fixed_n_pred=args.fixed_n_pred,
        subtitle=subtitle_b,
    )

    # =========================
    # Benchmark C: training time only vs training size
    # =========================
    print("\n=== Benchmark C: training time only vs training size ===")
    print(f"n_train grid: {train_grid}")

    fit_rows_all: List[Dict[str, Any]] = []
    for model_name, factory in icl_model_factories:
        print(f"\n--- {model_name} ---")
        rows_c = benchmark_train_time_vs_train_size(
            model_name=model_name,
            model_factory=factory,
            n_train_values=train_grid,
            n_test=args.n_test,
            n_features=args.n_features,
            seed=args.seed,
        )
        fit_rows_all.extend(rows_c)

        for r in rows_c:
            print(f"n_train={r['n_train']:>6} | fit={r['fit_time_s']:.3f}s | device={r['device']}")

    df_fit = pd.DataFrame(fit_rows_all)
    out_fit_csv = f"{base}_train_time_vs_train_size_v2.csv"
    df_fit.to_csv(out_fit_csv, index=False)
    print(f"\nSaved benchmark C results to: {out_fit_csv}")

    subtitle_c = f"fit only | n_features={args.n_features} | n_test={args.n_test}"
    plot_train_time_vs_train_size(
        df_fit=df_fit,
        base_out=base,
        subtitle=subtitle_c,
    )


if __name__ == "__main__":
    from mixture_hypernetworks import *
    main()
