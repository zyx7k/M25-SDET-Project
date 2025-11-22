#!/usr/bin/env python3

import time
from typing import Sequence

import jax
import jax.numpy as jnp
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, scale_y_log10,
    theme_minimal, labs, facet_wrap
)

# Import the functions / classes from your module
from simulation_jax import (
    init_position,
    simulate_signal,
    sin_signal,
    qpsk_signal,
    estimate_position,
    EstimationMethod,
    DefaultPosition
)

# ----------------------------
# Experiment configuration
# ----------------------------
NUM_RUNS = 100
# -20 dB ... +20 dB (step 2)
SNR_LIST = jnp.arange(-20, 21, 2)
GRID_MIN = 0.0
GRID_MAX = 10_000.0
GRID_STEP = 50.0

# Seeds for Monte-Carlo runs (jax-friendly)
SEED_START = 1000
SEEDS = jnp.arange(SEED_START, SEED_START + NUM_RUNS)

@jax.jit
def mse(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum((a - b) ** 2, axis=-1)

def single_run(seed: int, snr: float, pos: DefaultPosition, known_signal: bool):
    params = init_position(pos, seed, snr)
    signal_envelope = qpsk_signal(params.seed, params.num_timesteps, params.num_samples_per_interval)
    signal = simulate_signal(params, signal_envelope)

    if known_signal:
        ests, _ = estimate_position(
            params,
            signal,
            EstimationMethod.DirectPosition,
            GRID_MIN,
            GRID_MAX,
            GRID_STEP,
            prior_signal=signal_envelope
        )
    else:
        ests, _ = estimate_position(
            params,
            signal,
            EstimationMethod.DirectPosition,
            GRID_MIN,
            GRID_MAX,
            GRID_STEP,
            prior_signal=None
        )

    # if estimate_position returns multiple maxima, pick the one with the least mse
    mse_errors = mse(ests, params.emitter)
    return mse_errors.min()

# ----------------------------
# RMSE per SNR (vectorized over seeds)
# ----------------------------
def rmse_for_snr(snr: float, pos: DefaultPosition, known_signal: bool):
    rmse_vals = []
    for seed in SEEDS:
        rmse_vals.append(single_run(seed, snr, pos, known_signal))
    return jnp.sqrt(jnp.mean(jnp.array(rmse_vals)))

def run_scenario(pos: DefaultPosition, known_signal: bool):
    """
    Sweep SNR_LIST and compute RMSE at each SNR for the given scenario.
    Returns a list of RMSEs aligned with SNR_LIST.
    """
    out = []
    start = time.time()
    for snr in SNR_LIST.tolist():
        t0 = time.time()
        rmse_val = rmse_for_snr(float(snr), pos, known_signal)
        out.append(rmse_val)
        t1 = time.time()
        print(f"[{pos} {'known' if known_signal else 'unknown'}] SNR={snr} -> RMSE={rmse_val:.3f} m  (took {t1-t0:.2f}s)")
    end = time.time()
    print(f"Completed scenario {pos} known={known_signal} in {end-start:.2f}s")
    return out

def main():
    scenarios = [
        (DefaultPosition.PosA, False, "PosA Unknown (L=2)"),
        (DefaultPosition.PosA, True,  "PosA Known   (L=2)"),
        (DefaultPosition.PosB, False, "PosB Unknown (L=3)"),
        (DefaultPosition.PosB, True,  "PosB Known   (L=3)"),
    ]

    results = {}
    for pos, known, label in scenarios:
        print(f"\n=== Running scenario: {label} ===")
        vals = run_scenario(pos, known)
        results[label] = vals

    rows = []
    for label, vals in results.items():
        for snr, rmse_val in zip(SNR_LIST.tolist(), vals):
            rows.append({"SNR": float(snr), "RMSE": float(rmse_val), "Scenario": label})

    df = pd.DataFrame(rows)

    p = (
        ggplot(df, aes("SNR", "RMSE", color="Scenario"))
        + geom_line()
        + geom_point(size=1.5)
        + scale_y_log10()
        + facet_wrap("~Scenario", ncol=2, scales="free")
        + theme_minimal()
        + labs(title="RMSE vs SNR (100 Monte-Carlo runs)", x="SNR [dB]", y="RMSE [m]")
    )

    out_path = "jax_plots/rmse_plotnine.png"
    p.save(out_path, dpi=300, width=10, height=8)
    print(f"Saved plot to: {out_path}")

if __name__ == "__main__":
    main()
