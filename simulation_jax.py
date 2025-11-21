import sys
import enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Complex, Int
import pandas as pd
import plotnine as p9

# jax.config.update("jax_enable_x64", True)

SEED = 42

# Physical Constants
PROPOGATION_SPEED_VAL = 300_000_000.0
NOMINAL_CARRIER_FREQUENCY_VAL = 100_000_000.0
DEFAULT_RECEIVER_SPEED_VAL = 300.0

class EstimationMethod(enum.StrEnum):
    DirectPosition = "DPD"
    DifferentialDoppler = "DD"

@dataclass
class Params:
    seed: int
    rng: Array = field(init=False)

    num_receivers: int
    num_timesteps: int
    num_samples_per_interval: int
    sample_rate: int

    snr_ratio: float

    # JAX Arrays (Unitless SI)
    receivers_p: Float[Array, "2 timesteps receivers"]
    receivers_v: Float[Array, "2 timesteps receivers"]
    emitter: Float[Array, "2"] = field(init=False)
    timesteps: Float[Array, "timesteps"] = field(init=False)
    transmitted_freq_shifts: Float[Array, "receivers"] = field(init=False)
    channel_attenuation: Float[Array, "1"] = field(init=False)
    channel_phase: Float[Array, "1"] = field(init=False)


    def __post_init__(self):
        # check if we have the correct number
        req_shape = (2, self.num_timesteps, self.num_receivers)
        assert self.receivers_p.shape == req_shape
        assert self.receivers_v.shape == req_shape

        self.rng = jax.random.key(self.seed)
        self.rng, k1, k2, k3, k4, k5 = jax.random.split(self.rng, 6)

        # The emitter’s position is chosen at random within a square area of 10 x 10 [Km x Km].
        self.emitter = jnp.stack([
            jax.random.uniform(k1, minval=0.0, maxval=10_000.0),
            jax.random.uniform(k2, minval=0.0, maxval=10_000.0)
        ])

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
        v = self.receivers_v[..., 0]
        p = self.receivers_p[..., 0]

        timesteps: Float[Array, "timesteps receivers"] = jnp.where(v != 0, (p / v), -jnp.inf)

        self.timesteps = jnp.max(timesteps, axis=0)


class DefaultPosition(enum.StrEnum):
    PosA = 'A'
    PosB = 'B'
    PosC = 'C'
    PosD = 'D'
    PosTest = 'Test'

def init_position(position: DefaultPosition) -> Params:
    snr_ratio = 1.0
    sample_rate = 10_000

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
            snr_ratio = 0.0
            p_x = jnp.array([[0.0, 1000.0], [300.0, 1300.0], [600.0, 1600.0], [900.0, 1900.0]])
            p_y = jnp.zeros((T, L))
            v_x = jnp.full((T, L), 300.0, dtype=jnp.float32)
            v_y = jnp.zeros((T, L))

    N = 100

    return Params(
        seed=SEED,
        num_receivers=L, num_timesteps=T, num_samples_per_interval=N,
        sample_rate=sample_rate,
        snr_ratio=snr_ratio,
        receivers_p=jnp.stack([p_x, p_y]),
        receivers_v=jnp.stack([v_x, v_y])
    )

@jax.jit
def calculate_mu(
        p0: Float[Array, "2"],
        p_lk: Float[Array, "2 k l"],
        v_lk: Float[Array, "2 k l"],
        c: float
) -> Float[Array, "k l"]:
    p_diff = p0[:, None, None] - p_lk
    numerator = jnp.sum(v_lk * p_diff, axis=0)
    denominator = jnp.maximum(jnp.sqrt(jnp.sum(p_diff ** 2, axis=0)), 1e-9)
    return numerator / (c * denominator)

def sin_signal(k: int, N: int) -> Float[Array, "k N"]:
    k_vals = jnp.arange(k)
    n_vals = jnp.arange(N)
    return jnp.sin(2 * jnp.pi * jnp.outer(k_vals, n_vals) / (k * N) + 0.5)

def simulate_signal(params: Params, generate_signal: Callable[[int, int], Float[Array, "k n"]]) -> Complex[Array, "k l n"]:
    k, l, N = params.num_timesteps, params.num_receivers, params.num_samples_per_interval

    s: Float[Array, "k N"] = generate_signal(k, N)
    b: Complex[Array, "k l"] = params.channel_attenuation * jnp.exp(1j * params.channel_phase * jnp.ones((k, l)))

    mu: Float[Array, "k l"] = calculate_mu(
        params.emitter, params.receivers_p, params.receivers_v,
        PROPOGATION_SPEED_VAL
    )

    T_s = params.num_samples_per_interval / params.sample_rate
    exps: Float[Array, "N"] = jnp.arange(params.num_samples_per_interval) * T_s

    A: Complex[Array, "k l n"] = jnp.exp(1j * 2 * jnp.pi * NOMINAL_CARRIER_FREQUENCY_VAL * jnp.einsum('kl,n->kln', mu, exps))
    C: Complex[Array, "k n"] = jnp.exp(1j * 2 * jnp.pi * jnp.einsum('k,n->kn', params.transmitted_freq_shifts, exps))

    signal = (b[:, :, None] * A * C[:, None, :] * s[:, None, :])

    params.rng, k2 = jax.random.split(params.rng)
    signal_power = jnp.mean(jnp.abs(signal) ** 2)
    noise_variance = signal_power / (10 ** (params.snr_ratio / 10))
    noise_stddev = jnp.sqrt(noise_variance)

    w = jax.random.normal(k2, shape=(2, k, l, N)) * noise_stddev / 2
    w = w[0] + w[1] * 1.0j

    return signal + w

@jax.jit
def compute_cost_dd(
        p: Float[Array, "2"],
        signal: Complex[Array, "k l n"],
        sample_rate: float,
        receivers_p: Float[Array, "2 k l"],
        receivers_v: Float[Array, "2 k l"]
) -> Float[Array, "1"]:
    raise NotImplementedError("TODO")

@jax.jit
def compute_cost_unknown(
        p: Float[Array, "2"],
        signal: Complex[Array, "k l n"],
        exps: Float[Array, "n"],
        receivers_p: Float[Array, "2 k l"],
        receivers_v: Float[Array, "2 k l"]
) -> Float[Array, "1"]:
    mu: Float[Array, "k l"] = calculate_mu(p, receivers_p, receivers_v, PROPOGATION_SPEED_VAL)

    A: Complex[Array, "k l n"] = jnp.exp(1j * 2 * jnp.pi * NOMINAL_CARRIER_FREQUENCY_VAL * jnp.einsum('kl,n->kln', mu, exps))
    V: Complex[Array, "k l n"] = jnp.conj(A) * signal
    # We take VV^H, instead of V^HV, to reduce computation
    Q: Complex[Array, "k l l"] = jnp.einsum('kln,kmn->klm', V, jnp.conj(V))

    # since the matrix is hermitian, we will always get real eigenvalues
    cost: Float[Array, "k"] = jnp.linalg.eigvalsh(Q)[:, -1]
    return jnp.sum(cost)

@jax.jit
def compute_cost_known(
        p: Float[Array, "2"],
        signal: Complex[Array, "k l n"],
        prior_signal: Complex[Array, "k n"],
        exps: Float[Array, "n"],
        receivers_p: Float[Array, "2 k l"],
        receivers_v: Float[Array, "2 k l"]
) -> Float[Array, "1"]:
    mu: Float[Array, "k l"] = calculate_mu(p, receivers_p, receivers_v, PROPOGATION_SPEED_VAL)

    A: Complex[Array, "k l n"] = jnp.exp(1j * 2 * jnp.pi * NOMINAL_CARRIER_FREQUENCY_VAL * jnp.einsum('kl,n->kln', mu, exps))
    V: Complex[Array, "k l n"] = jnp.conj(A) * signal
    # correlation with prior
    B: Complex[Array, "k l n"] = V * prior_signal[:, None, :]

    # Calculate energy at different time-lags via FFT
    # FFT of Autocorrelation = |FFT(Signal)|^2 (Power Spectral Density)
    # This replaces the O(N^2) matrix construction
    B_fft: Complex[Array, "k l n"] = jnp.fft.fft(B, axis=2)
    psd: Float[Array, "k l n"] = jnp.real(B_fft * jnp.conj(B_fft))

    # Sum energy across receivers
    psd_sum: Float[Array, "k n"] = jnp.sum(psd, axis=1)

    # Find peak energy (best lag) for each timestep and sum
    return jnp.sum(jnp.max(psd_sum, axis=1))

CostFn = Callable[[Float[Array, "N_batch 2"]], Float[Array, "1"]]
def estimate_position(
    params: Params,
    signal: Array,
    estimation_method: EstimationMethod,
    p_min: float,
    p_max: float,
    p_step: float,
    generate_prior_signal: Optional[Callable] = None
) -> tuple[Float[Array, "2"], dict[str, Float[Array, "N"]]]:
    grid = jnp.mgrid[p_min:p_max:p_step, p_min:p_max:p_step].reshape(2, -1)
    num_points = grid.shape[1]

    T_s = params.num_samples_per_interval / params.sample_rate
    exps: Float[Array, "N"] = jnp.arange(params.num_samples_per_interval) * T_s

    # we reshape to batches to prevent XLA from unrolling everything into one massive graph
    # if the grid is huge. 1024 is a safe batch size for GPU.
    batch_size = 1024
    pad = (batch_size - (num_points % batch_size)) % batch_size
    total_padded = num_points + pad

    # Pad inputs to divisible size
    grid_padded: Float[Array, "2 Np"] = jnp.pad(grid, ((0,0), (0, pad)))

    # Reshape to [num_batches, batch_size]
    grid_batched: Float[Array, "2 N_batch batch"] = grid_padded.reshape(2, -1, batch_size)
    grid_batched:  Float[Array, "N_batch batch 2"] = grid_batched.transpose(1, 2, 0)

    match estimation_method, generate_prior_signal:
        case (EstimationMethod.DirectPosition, None):
            cost_fn: CostFn = jax.vmap(lambda p: compute_cost_unknown(p, signal, exps, params.receivers_p, params.receivers_v))
        case (EstimationMethod.DirectPosition, f):
            prior: Float[Array, "k n"]  = f(params.num_timesteps, params.num_samples_per_interval)
            cost_fn: CostFn = jax.vmap(lambda p: compute_cost_known(p, signal, prior, exps, params.receivers_p, params.receivers_v))
        case (EstimationMethod.DifferentialDoppler, _):
            cost_fn: CostFn = jax.vmap(lambda p: compute_cost_dd(p, signal, float(params.sample_rate), params.receivers_p, params.receivers_v))

    def run_batch(idx_batch: int):
        batch: Float[Array, "N_batch"] = grid_batched[idx_batch]

        # note: cost_fn is a vmap
        return cost_fn(batch)

    batch_indices: Int[Array, "N_batch"] = jnp.arange(grid_batched.shape[0])
    costs_batched = jax.lax.map(run_batch, batch_indices)

    costs: Float[Array, "N_batch batch"] = costs_batched.flatten()[:num_points]

    max_idx: Int[Array, "1"] = jnp.argmax(costs)
    est = grid[:, max_idx]

    data = {
        "xs": grid[0],
        "ys": grid[1],
        "cost": costs
    }

    return est, data

if __name__ == '__main__':
    import time
    for SEED in range(1000, 1010):
        for idx, pos in enumerate([DefaultPosition.PosA, DefaultPosition.PosB, DefaultPosition.PosC, DefaultPosition.PosD]):
            print(f"\n--- Simulation: {pos} ---")
            params = init_position(pos)
            print(f"Emitter Actual: ({params.emitter})")

            signal = simulate_signal(params, sin_signal)

            start = time.time()
            estimate, data = estimate_position(
                params,
                signal,
                EstimationMethod.DirectPosition,
                0.0,
                10_000.0,
                100.0, # 100x100 grid
                # generate_prior_signal=sin_signal
            )
            end = time.time()

            est = estimate
            print(f"Estimate: ({est[0]:.2f}, {est[1]:.2f})")
            err = jnp.sum(jnp.sqrt((params.emitter - est)**2))

            data = pd.DataFrame(data)
            (
                p9.ggplot(data, p9.aes("xs", "ys", fill="cost"))
                + p9.geom_tile()
                + p9.geom_vline(xintercept=params.emitter[0])
                + p9.geom_hline(yintercept=params.emitter[1])
                + p9.geom_vline(xintercept=est[0], linetype="dotted")
                + p9.geom_hline(yintercept=est[1], linetype="dotted")
                + p9.theme_minimal()
            ).save(f'jax_plots/{idx}_{SEED}.png', dpi=300, width=5, height=5)

            print(f"Error: {err} m (Computed in {(end-start) * 1000:.3f}ms)")
        break
