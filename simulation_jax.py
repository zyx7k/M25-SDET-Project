import sys
import enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Complex
import pandas as pd
import plotnine as p9

# jax.config.update("jax_enable_x64", True)

SEED = 42

# Physical Constants
PROPOGATION_SPEED_VAL = 300_000_000.0
NOMINAL_CARRIER_FREQUENCY_VAL = 100_000_000.0
DEFAULT_RECEIVER_SPEED_VAL = 300.0

@dataclass
class Params:
    seed: int
    rng: Array = field(init=False)

    num_receivers: int
    num_timesteps: int
    num_samples_per_interval: int
    noise_stddev: float
    
    # JAX Arrays (Unitless SI)
    receivers_p_x: Float[Array, "timesteps receivers"]
    receivers_p_y: Float[Array, "timesteps receivers"]
    receivers_v_x: Float[Array, "timesteps receivers"]
    receivers_v_y: Float[Array, "timesteps receivers"]
    emitter_x: Float[Array, "1"] = field(init=False)
    emitter_y: Float[Array, "1"] = field(init=False)
    timesteps: Float[Array, "timesteps"] = field(init=False)
    transmitted_freq_shifts: Float[Array, "receivers"] = field(init=False)
    channel_attenuation: Float[Array, "1"] = field(init=False)
    channel_phase: Float[Array, "1"] = field(init=False)


    def __post_init__(self):
        # check if we have the correct number
        req_shape = (self.num_timesteps, self.num_receivers)
        assert self.receivers_p_x.shape == req_shape
        assert self.receivers_p_y.shape == req_shape
        assert self.receivers_v_x.shape == req_shape
        assert self.receivers_v_y.shape == req_shape

        self.rng = jax.random.key(self.seed)
        self.rng, k1, k2, k3, k4, k5 = jax.random.split(self.rng, 6)

        # The emitter’s position is chosen at random within a square area of 10 x 10 [Km x Km].
        self.emitter_x = jax.random.uniform(k1, minval=0.0, maxval=10_000.0)
        self.emitter_y = jax.random.uniform(k2, minval=0.0, maxval=10_000.0)

        # The unknown transmitted frequency shifts, {\mathcal{v}_k}, are selected at random
        # from the interval [-100, 100] Hz
        self.transmitted_freq_shifts = jax.random.uniform(k3, (self.num_timesteps,), minval=-100.0, maxval=100.0)
        # The channel attenuation is selected at random from a normal distribution with
        # mean one and standard deviation 0.1
        self.channel_attenuation = 1 + 0.1 * jax.random.normal(k4)
        # and the channel phase is selected at random from a
        # uniform distribution over [-pi, pi]
        self.channel_phase = jax.random.uniform(k5, minval=-jnp.pi, maxval=jnp.pi)

        # we assume that the velocity vector remains constant between each interception interval
        # we also assume that the interception times are the same for all receivers
        # therefore, we can calculate the time simply by taking the max time along x and y
        # (just incase any direction is 0)
        vx, vy = self.receivers_v_x[:, 0], self.receivers_v_y[:, 0]
        px, py = self.receivers_p_x[:, 0], self.receivers_p_y[:, 0]

        timesteps_x: Float[Array, "timesteps receivers"] = jnp.where(vx != 0, (px / vx), 0.0)
        timesteps_y: Float[Array, "timesteps receivers"] = jnp.where(vy != 0, (py / vy), 0.0)

        self.timesteps = jnp.maximum(timesteps_x, timesteps_y)


class DefaultPosition(enum.StrEnum):
    PosA = 'A'
    PosB = 'B'
    PosC = 'C'
    PosD = 'D'
    PosTest = 'Test'

def init_position(position: DefaultPosition) -> Params:
    noise_std = 1.0
    
    match position:
        case DefaultPosition.PosA:
            L, T = 2, 10
            p_x = jnp.stack([jnp.arange(1, 11), jnp.arange(10, 0, -1)]).T * 1000.0
            p_y = jnp.stack([jnp.full(T, 0.0), jnp.full(T, 10.0)]).T * 1000.0
            v_x = jnp.full((T, L), DEFAULT_RECEIVER_SPEED_VAL)
            v_y = jnp.zeros((T, L), dtype=jnp.float32)
        case DefaultPosition.PosB:
            L, T = 3, 10
            p_x = jnp.stack([jnp.arange(1, 11), jnp.arange(10, 0, -1), jnp.arange(1, 11)]).T * 1000
            p_y = jnp.stack([jnp.full(T, 0.0), jnp.full(T, 10.0), jnp.full(T, 0.2)]).T * 1000
            v_x = jnp.full((T, L), DEFAULT_RECEIVER_SPEED_VAL)
            v_y = jnp.zeros((T, L), dtype=jnp.float32)
        case DefaultPosition.PosC:
            L, T = 2, 10
            p_x = jnp.stack([jnp.arange(1, 11), jnp.full(T, 10.0)]).T * 1000
            p_y = jnp.stack([jnp.full(T, 0.0), jnp.arange(1, 11)]).T * 1000
            v_x = jnp.stack([jnp.full(T, DEFAULT_RECEIVER_SPEED_VAL), jnp.zeros(T)]).T
            v_y = jnp.stack([jnp.zeros(T), jnp.full(T, DEFAULT_RECEIVER_SPEED_VAL)]).T
        case DefaultPosition.PosD:
            L, T = 2, 10
            p_x = jnp.tile(jnp.arange(1, 11), (L, 1)).T * 1000
            p_y = jnp.stack([jnp.full(T, 0.0), jnp.full(T, -0.5)]).T * 1000
            v_x = jnp.full((T, L), DEFAULT_RECEIVER_SPEED_VAL)
            v_y = jnp.zeros((T, L))
        case DefaultPosition.PosTest:
            L, T = 2, 4
            noise_std = 0.0
            p_x = jnp.array([[0.0, 1000.0], [300.0, 1300.0], [600.0, 1600.0], [900.0, 1900.0]])
            p_y = jnp.zeros((T, L))
            v_x = jnp.full((T, L), 300.0, dtype=jnp.float32)
            v_y = jnp.zeros((T, L))

    N = 100

    return Params(
        seed=SEED,
        num_receivers=L, num_timesteps=T, num_samples_per_interval=N,
        noise_stddev=noise_std,
        receivers_p_x=p_x,
        receivers_p_y=p_y,
        receivers_v_x=v_x,
        receivers_v_y=v_y,
    )

@jax.jit
def calculate_mu(
        p0_x: Float[Array, "1"],
        p0_y: Float[Array, "1"],
        p_lk_x: Float[Array, "k l"],
        p_lk_y: Float[Array, "k l"],
        v_lk_x: Float[Array, "k l"],
        v_lk_y: Float[Array, "k l"],
        c: float
) -> Float[Array, "k l"]:
    p_diff_x = p0_x - p_lk_x
    p_diff_y = p0_y - p_lk_y
    numerator = v_lk_x * p_diff_x + v_lk_y * p_diff_y
    denominator = jnp.maximum(jnp.sqrt(p_diff_x ** 2 + p_diff_y ** 2), 1e-9)
    return numerator / (c * denominator)

def sin_signal(k: int, N: int) -> Float[Array, "k N"]:
    k_vals = jnp.arange(k)
    n_vals = jnp.arange(N)
    return jnp.sin(2 * jnp.pi * jnp.outer(k_vals, n_vals) / (k * N) + 0.5)

def simulate_signal(params: Params, generate_signal: Callable[[int, int], Float[Array, "k n"]]) -> Array:
    key = jax.random.PRNGKey(params.seed)
    k, l, N = params.num_timesteps, params.num_receivers, params.num_samples_per_interval
    
    s: Float[Array, "k N"] = generate_signal(k, N)
    b: Float[Array, "k l"] = jnp.ones((k, l))
    w: Float[Array, "k l N"] = jax.random.normal(key, (k, l, N)) * params.noise_stddev

    mu: Float[Array, "k l"] = calculate_mu(
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

@jax.jit
def compute_cost_unknown(p_x, p_y, signal, exps, receivers_p_x, receivers_p_y, receivers_v_x, receivers_v_y):
    mu = calculate_mu(p_x, p_y, receivers_p_x, receivers_p_y, receivers_v_x, receivers_v_y, PROPOGATION_SPEED_VAL)
    
    A = jnp.exp(1j * 2 * jnp.pi * NOMINAL_CARRIER_FREQUENCY_VAL * jnp.einsum('kl,n->kln', mu, exps))
    V = jnp.conj(A) * signal
    Q_k = jnp.einsum('kln,kmn->klm', V, jnp.conj(V))
    
    eigvals = jnp.linalg.eigvalsh(Q_k)
    return jnp.sum(eigvals[:, -1]).real

@jax.jit
def compute_cost_known(p_x, p_y, signal, prior_signal, exps, receivers_p_x, receivers_p_y, receivers_v_x, receivers_v_y):
    mu = calculate_mu(p_x, p_y, receivers_p_x, receivers_p_y, receivers_v_x, receivers_v_y, PROPOGATION_SPEED_VAL)
    
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
    p_min: float, 
    p_max: float, 
    p_step: float, 
    generate_prior_signal: Optional[Callable] = None
) -> tuple[tuple[Float[Array, "1"], Float[Array, "1"]], dict[str, Float[Array, "N"]]]:
    grid = jnp.mgrid[p_min:p_max:p_step, p_min:p_max:p_step].reshape(2, -1)
    flat_xs = grid[0, :]
    flat_ys = grid[1, :]
    num_points = flat_xs.shape[0]

    T_s = 100.0 / 10000.0
    exps = jnp.arange(params.num_samples_per_interval) * T_s
    
    # we reshape to batches to prevent XLA from unrolling everything into one massive graph
    # if the grid is huge. 1024 is a safe batch size for GPU.
    batch_size = 1024
    pad = (batch_size - (num_points % batch_size)) % batch_size
    total_padded = num_points + pad
    
    # Pad inputs to divisible size
    xs_padded = jnp.pad(flat_xs, (0, pad))
    ys_padded = jnp.pad(flat_ys, (0, pad))
    
    # Reshape to [num_batches, batch_size]
    xs_batched = xs_padded.reshape(-1, batch_size)
    ys_batched = ys_padded.reshape(-1, batch_size)

    # Run loop on device using lax.map
    def run_batch(idx_batch: int):
        # We actually need to pass the coordinates, scan_fn above used global flat_xs reference
        # Let's redefine to be cleaner
        b_xs = xs_batched[idx_batch]
        b_ys = ys_batched[idx_batch]
        if generate_prior_signal is None:
            return jax.vmap(lambda x, y: compute_cost_unknown(x, y, signal, exps, params.receivers_p_x, params.receivers_p_y, params.receivers_v_x, params.receivers_v_y))(b_xs, b_ys)
        else:
            prior = generate_prior_signal(params.num_timesteps, params.num_samples_per_interval)
            return jax.vmap(lambda x, y: compute_cost_known(x, y, signal, prior, exps, params.receivers_p_x, params.receivers_p_y, params.receivers_v_x, params.receivers_v_y))(b_xs, b_ys)

    # Use jax.lax.map to loop over batches
    # range is just indices 0..num_batches
    batch_indices = jnp.arange(xs_batched.shape[0])
    costs_batched = jax.lax.map(run_batch, batch_indices)
    
    # Flatten and slice off padding
    costs = costs_batched.flatten()[:num_points]
    
    max_idx = jnp.argmax(costs)
    best_x = flat_xs[max_idx]
    best_y = flat_ys[max_idx]

    data = {
        "xs": flat_xs,
        "ys": flat_ys,
        "cost": costs
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
                0.0, 
                10_000.0, 
                100.0, # 100x100 grid
                generate_prior_signal=sin_signal
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
                + p9.geom_vline(xintercept=est_x, linetype="dotted")
                + p9.geom_hline(yintercept=est_y, linetype="dotted")
                + p9.theme_minimal()
            ).save(f'jax_plots/__{idx}_{SEED}.png', dpi=300, width=5, height=5)
            
            print(f"Error: {err:.2f} m (Computed in {end-start:.2f}s)")
        break
