import sys
import enum
from dataclasses import dataclass
from typing import Optional, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Complex
import numpy as np
import pint
import pandas as pd
import plotnine as p9

# jax.config.update("jax_enable_x64", True)

SEED = 42

ureg = pint.UnitRegistry(force_ndarray=True)

# Physical Constants
PROPOGATION_SPEED_VAL = 300_000_000.0
NOMINAL_CARRIER_FREQUENCY_VAL = 100_000_000.0
DEFAULT_RECEIVER_SPEED_VAL = 300.0

@dataclass
class Params:
    seed: int
    num_receivers: int
    num_timesteps: int
    num_samples_per_interval: int
    noise_stddev: float
    
    # JAX Arrays (Unitless SI)
    receivers_p_x: Float[Array, "timesteps receivers"]
    receivers_p_y: Float[Array, "timesteps receivers"]
    receivers_v_x: Float[Array, "timesteps receivers"]
    receivers_v_y: Float[Array, "timesteps receivers"]
    emitter_x: float
    emitter_y: float
    timesteps: Float[Array, "timesteps"]
    transmitted_freq_shifts: Float[Array, "receivers"]
    channel_attenuation: float
    channel_phase: float

# Register PyTree
jax.tree_util.register_pytree_node(
    Params,
    lambda p: ((p.receivers_p_x, p.receivers_p_y, p.receivers_v_x, p.receivers_v_y, 
                p.emitter_x, p.emitter_y, p.timesteps, p.transmitted_freq_shifts,
                p.channel_attenuation, p.channel_phase), 
               (p.seed, p.num_receivers, p.num_timesteps, p.num_samples_per_interval, p.noise_stddev)),
    lambda aux, children: Params(
        seed=aux[0], num_receivers=aux[1], num_timesteps=aux[2], 
        num_samples_per_interval=aux[3], noise_stddev=aux[4],
        receivers_p_x=children[0], receivers_p_y=children[1], receivers_v_x=children[2],
        receivers_v_y=children[3], emitter_x=children[4], emitter_y=children[5],
        timesteps=children[6], transmitted_freq_shifts=children[7],
        channel_attenuation=children[8], channel_phase=children[9]
    )
)

class DefaultPosition(enum.StrEnum):
    PosA = 'A'
    PosB = 'B'
    PosC = 'C'
    PosD = 'D'
    PosTest = 'Test'

def init_position(position: DefaultPosition) -> Params:
    km = ureg.kilometer
    mps = ureg.meter_per_second
    m = ureg.meter
    
    rng = np.random.default_rng(SEED)
    noise_std = 1.0
    
    match position:
        case DefaultPosition.PosA:
            L, T = 2, 10
            p_x = np.stack([np.arange(1, 11), np.arange(10, 0, -1)]).T * km
            p_y = np.stack([np.full(T, 0.0), np.full(T, 10.0)]).T * km
            v_x = np.full((T, L), DEFAULT_RECEIVER_SPEED_VAL) * mps
            v_y = np.zeros((T, L)) * mps
        case DefaultPosition.PosB:
            L, T = 3, 10
            p_x = np.stack([np.arange(1, 11), np.arange(10, 0, -1), np.arange(1, 11)]).T * km
            p_y = np.stack([np.full(T, 0.0), np.full(T, 10.0), np.full(T, 0.2)]).T * km
            v_x = np.full((T, L), DEFAULT_RECEIVER_SPEED_VAL) * mps
            v_y = np.zeros((T, L)) * mps
        case DefaultPosition.PosC:
            L, T = 2, 10
            p_x = np.stack([np.arange(1, 11), np.full(T, 10.0)]).T * km
            p_y = np.stack([np.full(T, 0.0), np.arange(1, 11)]).T * km
            v_x = np.stack([np.full(T, DEFAULT_RECEIVER_SPEED_VAL), np.zeros(T)]).T * mps
            v_y = np.stack([np.zeros(T), np.full(T, DEFAULT_RECEIVER_SPEED_VAL)]).T * mps
        case DefaultPosition.PosD:
            L, T = 2, 10
            p_x = np.tile(np.arange(1, 11), (L, 1)).T * km
            p_y = np.stack([np.full(T, 0.0), np.full(T, -0.5)]).T * km
            v_x = np.full((T, L), DEFAULT_RECEIVER_SPEED_VAL) * mps
            v_y = np.zeros((T, L)) * mps
        case DefaultPosition.PosTest:
            L, T = 2, 4
            noise_std = 0.0
            p_x = np.array([[0.0, 1000.0], [300.0, 1300.0], [600.0, 1600.0], [900.0, 1900.0]]) * m
            p_y = np.zeros((T, L)) * m
            v_x = np.full((T, L), 300) * mps
            v_y = np.zeros((T, L)) * mps

    N = 100 if position != DefaultPosition.PosTest else 128

    if position == DefaultPosition.PosTest:
        emitter_x_val, emitter_y_val = 400.0, 300.0
        shifts = np.zeros(T)
        attenuation, phase = 1.0, 0.0
    else:
        emitter_x_val = rng.uniform(0, 10_000) 
        emitter_y_val = rng.uniform(0, 10_000)
        shifts = rng.uniform(-100, 100, T)
        attenuation = rng.normal(1, 0.1)
        phase = rng.uniform(-np.pi, np.pi)

    # Calculate Timesteps
    vx_m, vy_m = v_x[:, 0].to('m/s').m, v_y[:, 0].to('m/s').m
    px_m, py_m = p_x[:, 0].to('m').m, p_y[:, 0].to('m').m
    
    with np.errstate(divide='ignore', invalid='ignore'):
        t_x = np.where(vx_m != 0, px_m / vx_m, 0.0)
        t_y = np.where(vy_m != 0, py_m / vy_m, 0.0)
    timesteps_val = np.maximum(t_x, t_y)

    return Params(
        seed=SEED,
        num_receivers=L, num_timesteps=T, num_samples_per_interval=N,
        noise_stddev=noise_std,
        receivers_p_x=jnp.array(p_x.to('m').m),
        receivers_p_y=jnp.array(p_y.to('m').m),
        receivers_v_x=jnp.array(v_x.to('m/s').m),
        receivers_v_y=jnp.array(v_y.to('m/s').m),
        emitter_x=jnp.array(emitter_x_val),
        emitter_y=jnp.array(emitter_y_val),
        timesteps=jnp.array(timesteps_val),
        transmitted_freq_shifts=jnp.array(shifts),
        channel_attenuation=jnp.array(attenuation),
        channel_phase=jnp.array(phase)
    )

@jax.jit
def calculate_mu(p0_x, p0_y, p_lk_x, p_lk_y, v_lk_x, v_lk_y, c):
    p_diff_x = p0_x - p_lk_x
    p_diff_y = p0_y - p_lk_y
    numerator = v_lk_x * p_diff_x + v_lk_y * p_diff_y
    denominator = jnp.maximum(jnp.sqrt(p_diff_x ** 2 + p_diff_y ** 2), 1e-9)
    return numerator / (c * denominator)

def sin_signal(k: int, N: int) -> Array:
    k_vals = jnp.arange(k)
    n_vals = jnp.arange(N)
    return jnp.sin(2 * jnp.pi * jnp.outer(k_vals, n_vals) / (k * N) + 0.5)

def simulate_signal(params: Params, generate_signal: Callable[[int, int], Float[Array, "k n"]]) -> Array:
    key = jax.random.PRNGKey(params.seed)
    k, l, N = params.num_timesteps, params.num_receivers, params.num_samples_per_interval
    
    s = generate_signal(k, N)
    b = jnp.ones((k, l))
    w = jax.random.normal(key, (k, l, N)) * params.noise_stddev

    mu = calculate_mu(
        params.emitter_x, params.emitter_y,
        params.receivers_p_x, params.receivers_p_y,
        params.receivers_v_x, params.receivers_v_y,
        PROPOGATION_SPEED_VAL
    )

    T_s = 100.0 / 10000.0 
    exps = jnp.arange(N) * T_s

    A = jnp.exp(1j * 2 * jnp.pi * NOMINAL_CARRIER_FREQUENCY_VAL * jnp.einsum('kl,n->kln', mu, exps))
    C = jnp.exp(1j * 2 * jnp.pi * jnp.einsum('k,n->kn', params.transmitted_freq_shifts, exps))
    
    return (b[:, :, None] * A * C[:, None, :] * s[:, None, :]) + w

# --- Optimized Estimation Kernels ---

@jax.jit
def compute_cost_unknown(p_x, p_y, signal, params):
    # Unknown signal: Uses eigenvalues of the covariance matrix
    mu = calculate_mu(p_x, p_y, params.receivers_p_x, params.receivers_p_y, params.receivers_v_x, params.receivers_v_y, PROPOGATION_SPEED_VAL)
    T_s = 100.0 / 10000.0
    exps = jnp.arange(params.num_samples_per_interval) * T_s
    
    A = jnp.exp(1j * 2 * jnp.pi * NOMINAL_CARRIER_FREQUENCY_VAL * jnp.einsum('kl,n->kln', mu, exps))
    V = jnp.conj(A) * signal
    
    # Compute Q matrices [K, L, L] by contracting over N
    # This is fast for small L
    Q_k = jnp.einsum('kln,kmn->klm', V, jnp.conj(V))
    
    # Sum of max eigenvalues over timesteps K
    eigvals = jnp.linalg.eigvalsh(Q_k)
    return jnp.sum(eigvals[:, -1]).real

@jax.jit
def compute_cost_known(p_x, p_y, signal, prior_signal, params):
    # Known signal: Uses FFT-based correlation (Wiener-Khinchin Theorem)
    # O(N log N) instead of O(N^2)
    mu = calculate_mu(p_x, p_y, params.receivers_p_x, params.receivers_p_y, params.receivers_v_x, params.receivers_v_y, PROPOGATION_SPEED_VAL)
    T_s = 100.0 / 10000.0
    exps = jnp.arange(params.num_samples_per_interval) * T_s
    
    # Demodulate Doppler shift
    A = jnp.exp(1j * 2 * jnp.pi * NOMINAL_CARRIER_FREQUENCY_VAL * jnp.einsum('kl,n->kln', mu, exps))
    
    # V: [K, L, N] - Signal compensated for position-dependent Doppler
    V = jnp.conj(A) * signal
    
    # B: [K, L, N] - Match with prior signal
    B = V * prior_signal[:, None, :]
    
    # Calculate energy at different time-lags via FFT
    # FFT of Autocorrelation = |FFT(Signal)|^2 (Power Spectral Density)
    # This replaces the O(N^2) matrix construction
    B_fft = jnp.fft.fft(B, axis=2)
    psd = jnp.real(B_fft * jnp.conj(B_fft))
    
    # Sum energy across receivers [K, L, N] -> [K, N]
    psd_sum = jnp.sum(psd, axis=1)
    
    # Find peak energy (best lag) for each timestep and sum
    return jnp.sum(jnp.max(psd_sum, axis=1))

def estimate_direct_position(
    params: Params, 
    signal: Array, 
    p_min: pint.Quantity, 
    p_max: pint.Quantity, 
    p_step: pint.Quantity, 
    generate_prior_signal: Optional[Callable] = None
) -> Tuple[Tuple[float, float], dict]:
    
    # 1. Create Grid
    xs = np.arange(p_min.to('m').magnitude, p_max.to('m').magnitude, p_step.to('m').magnitude)
    ys = np.arange(p_min.to('m').magnitude, p_max.to('m').magnitude, p_step.to('m').magnitude)
    grid_xs, grid_ys = np.meshgrid(xs, ys)
    
    # Shape: [Num_Points]
    flat_xs = jnp.array(grid_xs.ravel())
    flat_ys = jnp.array(grid_ys.ravel())
    num_points = flat_xs.shape[0]
    
    # 2. Prepare Scanning Function
    if generate_prior_signal is None:
        def scan_fn(idx):
            return compute_cost_unknown(flat_xs[idx], flat_ys[idx], signal, params)
    else:
        prior = generate_prior_signal(params.num_timesteps, params.num_samples_per_interval)
        def scan_fn(idx):
            return compute_cost_known(flat_xs[idx], flat_ys[idx], signal, prior, params)

    # 3. Execution with Batching via jax.lax.map
    # Instead of Python looping, we map the function over indices directly on device.
    # Chunking is handled automatically by map/scan logic usually, but if VRAM is tight,
    # we can reshape to explicit batches. 
    
    print(f"Computing cost for {num_points} grid points (Optimized FFT)...")
    
    # We reshape to batches to prevent XLA from unrolling everything into one massive graph
    # if the grid is huge. 1024 is a safe batch size for GPU.
    batch_size = 4096
    pad = (batch_size - (num_points % batch_size)) % batch_size
    total_padded = num_points + pad
    
    # Pad inputs to divisible size
    xs_padded = jnp.pad(flat_xs, (0, pad))
    ys_padded = jnp.pad(flat_ys, (0, pad))
    
    # Reshape to [num_batches, batch_size]
    xs_batched = xs_padded.reshape(-1, batch_size)
    ys_batched = ys_padded.reshape(-1, batch_size)
    
    # Define batched kernel
    batched_kernel = jax.vmap(scan_fn) # Vectorizes over batch dimension
    
    # Run loop on device using lax.map
    def run_batch(idx_batch):
        # We actually need to pass the coordinates, scan_fn above used global flat_xs reference
        # Let's redefine to be cleaner
        b_xs = xs_batched[idx_batch]
        b_ys = ys_batched[idx_batch]
        if generate_prior_signal is None:
            return jax.vmap(lambda x, y: compute_cost_unknown(x, y, signal, params))(b_xs, b_ys)
        else:
            prior = generate_prior_signal(params.num_timesteps, params.num_samples_per_interval)
            return jax.vmap(lambda x, y: compute_cost_known(x, y, signal, prior, params))(b_xs, b_ys)

    # Use jax.lax.map to loop over batches
    # range is just indices 0..num_batches
    batch_indices = jnp.arange(xs_batched.shape[0])
    costs_batched = jax.lax.map(run_batch, batch_indices)
    
    # Flatten and slice off padding
    costs = costs_batched.flatten()[:num_points]
    
    # Move to host
    costs_np = np.array(costs)
    flat_xs_np = np.array(flat_xs)
    flat_ys_np = np.array(flat_ys)
    
    max_idx = np.argmax(costs_np)
    best_x = flat_xs_np[max_idx]
    best_y = flat_ys_np[max_idx]

    data = {
        "xs": flat_xs_np,
        "ys": flat_ys_np,
        "cost": costs_np
    }
    
    return (best_x, best_y), data

if __name__ == '__main__':
    import time
    for SEED in range(1000, 1010):
        for idx, pos in enumerate([DefaultPosition.PosA, DefaultPosition.PosB, DefaultPosition.PosC, DefaultPosition.PosD]):
            print(f"\n--- Simulation: {pos} ---")
            params = init_position(pos)
            print(f"Emitter Actual: ({params.emitter_x:.2f}, {params.emitter_y:.2f})")
            
            signal = simulate_signal(params, sin_signal)
            
            start = time.time()
            estimate, data = estimate_direct_position(
                params, 
                signal, 
                0 * ureg.km, 
                10 * ureg.km, 
                0.01 * ureg.km, # 100x100 grid
                # generate_prior_signal=sin_signal
            )
            end = time.time()
            
            est_x, est_y = estimate
            print(f"Estimate: ({est_x:.2f}, {est_y:.2f})")
            
            err = np.sqrt((params.emitter_x - est_x)**2 + (params.emitter_y - est_y)**2)

            data = pd.DataFrame(data)
            (
                p9.ggplot(data, p9.aes("xs", "ys", fill="cost"))
                + p9.geom_tile()
                + p9.geom_vline(xintercept=params.emitter_x)
                + p9.geom_hline(yintercept=params.emitter_y)
                + p9.theme_minimal()
            ).save(f'jax_plots/__{idx}_{SEED}.png', dpi=300, width=5, height=5)
            
            print(f"Error: {err:.2f} m (Computed in {end-start:.2f}s)")
        break
