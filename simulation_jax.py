import sys
import enum
from dataclasses import dataclass, field, replace
from typing import Optional, Callable, Tuple, Self

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import Array
from jaxtyping import Float, Complex, Int
import pandas as pd
import plotnine as p9

jax.config.update("jax_enable_x64", True)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_kernel_cache_file")

SEED = 42

# Physical Constants
PROPOGATION_SPEED_VAL = 300_000_000.0
NOMINAL_CARRIER_FREQUENCY_VAL = 100_000_000.0
DEFAULT_RECEIVER_SPEED_VAL = 300.0

class EstimationMethod(enum.StrEnum):
    DirectPosition = "DPD"
    DifferentialDoppler = "DD"

@jax.tree_util.register_pytree_node_class
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

    # jax utility functions for easy grad calculation
    def tree_flatten(self):
        children = (self.emitter, self.transmitted_freq_shifts, self.channel_attenuation, self.channel_phase)
        aux = (self.seed, self.rng, self.num_receivers, self.num_samples_per_interval, self.num_timesteps, self.sample_rate, self.snr_ratio, self.receivers_p, self.receivers_v, self.timesteps)

        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            emitter,
            transmitted_freq_shifts,
            channel_attenuation,
            channel_phase,
        ) = children
        (
            seed,
            rng,
            num_receivers,
            num_samples_per_interval,
            num_timesteps,
            sample_rate,
            snr_ratio,
            receivers_p,
            receivers_v,
            timesteps,
        ) = aux

        # Construct without calling __init__ or __post_init__
        self = cls.__new__(cls)

        # Manually set attributes
        self.snr_ratio = snr_ratio
        self.emitter = emitter
        self.receivers_p = receivers_p
        self.receivers_v = receivers_v
        self.timesteps = timesteps
        self.transmitted_freq_shifts = transmitted_freq_shifts
        self.channel_attenuation = channel_attenuation
        self.channel_phase = channel_phase

        self.seed = seed
        self.rng = rng
        self.num_receivers = num_receivers
        self.num_samples_per_interval = num_samples_per_interval
        self.num_timesteps = num_timesteps
        self.sample_rate = sample_rate

        return self

    @classmethod
    def copy(cls, params: Self, seed: Optional[int] = None, snr_ratio: Optional[float] = None):
        if seed is None:
            seed = params.seed
        if snr_ratio is None:
            snr_ratio = params.snr_ratio

        return replace(params, seed=seed, snr_ratio=snr_ratio)

class DefaultPosition(enum.StrEnum):
    PosA = 'A'
    PosB = 'B'
    PosC = 'C'
    PosD = 'D'
    PosTest = 'Test'

def init_position(position: DefaultPosition, seed: int = 42, snr_ratio: float = 1.0) -> Params:
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

    N = 128

    return Params(
        seed=seed,
        num_receivers=L, num_timesteps=T, num_samples_per_interval=N,
        sample_rate=sample_rate,
        snr_ratio=snr_ratio,
        receivers_p=jnp.stack([p_x, p_y], dtype=jnp.float32),
        receivers_v=jnp.stack([v_x, v_y], dtype=jnp.float32)
    )

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

def sin_signal(seed: int, k: int, N: int) -> Float[Array, "k N"]:
    key = jax.random.key(seed)
    bits = jax.random.uniform(key, shape=(k, N), minval=-jnp.pi, maxval=jnp.pi)
    return jnp.sin(2 * bits / (k * N) + 0.5)

    # k_vals = jnp.arange(k)
    # n_vals = jnp.arange(N)
    # return jnp.sin(2 * jnp.pi * jnp.outer(k_vals, n_vals) / (k * N) + 0.5)

def qpsk_signal(seed: int, k: int, N: int) -> Float[Array, "k N"]:
    """
    Generate k independent QPSK baseband signals of length N symbols each.
    Returns complex64 array with shape (k, N).

    Symbol rate = 10 kbps is implied (each symbol carries 2 bits),
    but since this is baseband complex QPSK, no explicit sampling
    rate is needed unless you want oversampling.
    """
    key = jax.random.key(seed)
    bits = jax.random.randint(key, shape=(k, N, 2), minval=0, maxval=2)

    @jax.jit
    def gen_one(bits):
        # Map bits → QPSK constellation (Gray code)
        # 00 → +1 +1j
        # 01 → -1 +1j
        # 11 → -1 -1j
        # 10 → +1 -1j
        I = 1 - 2 * bits[:, 0]
        Q = 1 - 2 * bits[:, 1]

        # Normalize power to 1
        s = (I + 1j * Q) / jnp.sqrt(2.0)
        return s.astype(jnp.complex64)

    return jax.vmap(gen_one, in_axes=0)(bits)

def simulate_noisefree_signal(params: Params, s: Complex[Array, "k n"]) -> Complex[Array, "k l n"]:
    k, l, N = params.num_timesteps, params.num_receivers, params.num_samples_per_interval

    b: Complex[Array, "k l"] = params.channel_attenuation * jnp.exp(1j * params.channel_phase * jnp.ones((k, l)))

    T_s = 1.0 / params.sample_rate
    exps: Float[Array, "N"] = jnp.arange(params.num_samples_per_interval) * T_s

    mu: Float[Array, "k l"] = calculate_mu(params.emitter, params.receivers_p, params.receivers_v, PROPOGATION_SPEED_VAL)
    A: Complex[Array, "k l n"] = jnp.exp(1j * 2 * jnp.pi * NOMINAL_CARRIER_FREQUENCY_VAL * jnp.einsum('kl,n->kln', mu, exps))
    C: Complex[Array, "k n"] = jnp.exp(1j * 2 * jnp.pi * jnp.einsum('k,n->kn', params.transmitted_freq_shifts, exps))
    signal = (b[:, :, None] * A * C[:, None, :] * s[:, None, :])

    return signal

@jax.jit
def simulate_signal(params: Params, s: Complex[Array, "k n"]) -> Complex[Array, "k l n"]:
    k, l, N = params.num_timesteps, params.num_receivers, params.num_samples_per_interval

    signal = simulate_noisefree_signal(params, s)

    params.rng, k2 = jax.random.split(params.rng)
    signal_power = jnp.mean(jnp.abs(signal) ** 2)
    noise_variance = signal_power / (10 ** (params.snr_ratio / 10))
    noise_stddev = jnp.sqrt(noise_variance)

    w = jax.random.normal(k2, shape=(2, k, l, N)) * noise_stddev / jnp.sqrt(2)
    w = w[0] + w[1] * 1.0j

    return (signal + w)


def calculate_crlb(params: Params, signal: Complex[Array, "k N"], known=False, num_samples=100):
    def wrap_simulate_noisefree_signal(args):
        params_recon, signal_real, signal_imag = unravel_fn(args)
        out = simulate_noisefree_signal(params_recon, signal_real + 1.0j * signal_imag)
        return (jnp.real(out), jnp.imag(out))

    def sigma2_of_theta(th):
        p_recon, s_r, s_i = unravel_fn(th)
        mu_t = simulate_noisefree_signal(p_recon, s_r + 1.0j * s_i)
        pw = jnp.mean(jnp.abs(mu_t) ** 2)
        snr_l = 10.0 ** (p_recon.snr_ratio / 10.0)
        return pw / snr_l

    seeds = jnp.arange(params.seed, params.seed + num_samples)

    sum_fim: Optional[Float[Array, "params params"]] = None

    for seed in seeds:
        curr_params = Params.copy(params, seed=int(seed))

        theta, unravel_fn = ravel_pytree((curr_params, jnp.real(signal), jnp.imag(signal)))
        num_params = theta.shape[0]

        (J_r, J_i) = jax.jacfwd(wrap_simulate_noisefree_signal)(theta)
        J_s = J_r + 1.0j * J_i
        J_s = J_s.reshape(-1, num_params)
        D = J_s.shape[0]

        sigma2, J_var = jax.value_and_grad(sigma2_of_theta)(theta)

        fim = 2.0 / sigma2 * jnp.real(jnp.conj(J_s).T @ J_s) + (D / (sigma2 ** 2)) * (J_var[:, None] @ J_var[None, :])
        sum_fim: Float[Array, "params params"] = fim if (sum_fim is None) else (sum_fim + fim)

    F_avg = jnp.real(sum_fim / float(num_samples))
    crlb = jnp.linalg.pinv(F_avg)
    # print((jnp.linalg.eigvals(crlb) < 1e-3).sum(), crlb.shape)
    # print((jnp.linalg.eigvals(F_avg) < 1e-3).sum(), F_avg.shape)

    # eigvals = jnp.linalg.eigvalsh(F_avg)
    # cond = jnp.max(jnp.abs(eigvals)) / jnp.maximum(jnp.min(jnp.abs(eigvals)), 1e-30)
    # print("F shape:", F_avg.shape)
    # print("eigvals (smallest 10):", eigvals[:10])
    # print("min eig, max eig, cond:", eigvals.min(), eigvals.max(), cond)

    crlb_emitter_x = jnp.real(crlb[0, 0])
    crlb_emitter_y = jnp.real(crlb[1, 1])

    return crlb_emitter_x, crlb_emitter_y, 1 / F_avg[0][0], 1 / F_avg[1][1]

@jax.jit
def compute_cost_dd(
        p: Float[Array, "2"],
        signal: Complex[Array, "k l n"],
        exps: Float[Array, "n"],
        receivers_p: Float[Array, "2 k l"],
        receivers_v: Float[Array, "2 k l"]
) -> Float[Array, "1"]:
    # todo: replace the 0/1s with all signals?
    r1 = 0
    r2 = 1

    dt = exps[1] - exps[0]
    N = exps.shape[0]

    S1: Float[Array, "k N"] = jnp.fft.fft(signal[:, r1, :], axis=-1)
    S2: Float[Array, "k N"] = jnp.fft.fft(signal[:, r2, :], axis=-1)
    cross_spec: Float[Array, "k N"] = S1 * jnp.conj(S2)

    mag: Float[Array, "k N"] = jnp.abs(cross_spec)
    peak_idx: Float[Array, "k"] = jnp.argmax(mag, axis=-1)
    freqs: Float[Array, "N"] = jnp.fft.fftfreq(N, d=dt)
    f_peak: Float[Array, "k"] = freqs[peak_idx]

    mu: Float[Array, "k l"] = calculate_mu(p, receivers_p, receivers_v, PROPOGATION_SPEED_VAL)
    pred_delta_f: Float[Array, "k"] = NOMINAL_CARRIER_FREQUENCY_VAL * (mu[:, r1] - mu[:, r2])

    err: Float[Array, "k"] = jnp.real(f_peak) - pred_delta_f
    cost = jnp.sum(err ** 2)
    return -cost

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
        prior_signal: Optional[Float[Array, "K N"]] = None
) -> tuple[Float[Array, "points 2"], dict[str, Float[Array, "N"]]]:
    grid = jnp.mgrid[p_min:p_max:p_step, p_min:p_max:p_step].reshape(2, -1)
    num_points = grid.shape[1]

    T_s = 1.0 / params.sample_rate
    exps: Float[Array, "N"] = jnp.arange(params.num_samples_per_interval) * T_s

    # we reshape to batches to prevent XLA from unrolling everything into one massive graph
    # if the grid is huge. 1024 is a safe batch size for GPU.
    batch_size = 64
    pad = (batch_size - (num_points % batch_size)) % batch_size
    total_padded = num_points + pad

    # Pad inputs to divisible size
    grid_padded: Float[Array, "2 Np"] = jnp.pad(grid, ((0,0), (0, pad)))

    # Reshape to [num_batches, batch_size]
    grid_batched: Float[Array, "2 N_batch batch"] = grid_padded.reshape(2, -1, batch_size)
    grid_batched:  Float[Array, "N_batch batch 2"] = grid_batched.transpose(1, 2, 0)

    match estimation_method, prior_signal:
        case (EstimationMethod.DirectPosition, None):
            cost_fn: CostFn = jax.vmap(lambda p: compute_cost_unknown(p, signal, exps, params.receivers_p, params.receivers_v))
        case (EstimationMethod.DirectPosition, f):
            cost_fn: CostFn = jax.vmap(lambda p: compute_cost_known(p, signal, prior_signal, exps, params.receivers_p, params.receivers_v))
        case (EstimationMethod.DifferentialDoppler, _):
            cost_fn: CostFn = jax.vmap(lambda p: compute_cost_dd(p, signal, exps, params.receivers_p, params.receivers_v))

    costs_batched = jax.lax.map(cost_fn, grid_batched)

    costs: Float[Array, "N_batch batch"] = costs_batched.flatten()[:num_points]

    max_cost: Float[Array, "points"] = jnp.max(costs)
    ests = grid.T[costs == max_cost]

    data = {
        "xs": grid[0],
        "ys": grid[1],
        "cost": costs
    }

    return ests, data


if __name__ == '__main__':
    import time

    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 1
    # jax.profiler.start_trace("/tmp/jax-trace", profiler_options=options)
    for seed in range(1000, 1010):
        for idx, pos in enumerate([DefaultPosition.PosA, DefaultPosition.PosB, DefaultPosition.PosC, DefaultPosition.PosD]):
            print(f"\n--- Simulation: {pos} ---")
            params = init_position(pos, seed=seed)
            print(f"Emitter Actual: ({params.emitter})")

            transmitted_signal = qpsk_signal(params.seed, params.num_timesteps, params.num_samples_per_interval)

            signal = simulate_signal(params, transmitted_signal)

            crlb = calculate_crlb(params, signal, known=False, num_samples=1)
            print(f"CRLB: {crlb}")

            start = time.time()
            estimate, data = estimate_position(
                params,
                signal,
                EstimationMethod.DirectPosition,
                0.0,
                10_000.0,
                100.0, # 100x100 grid
                prior_signal=transmitted_signal
            )

            end = time.time()

            print(f"All estimates: {estimate}")
            errs = jnp.sum(jnp.sqrt((params.emitter - estimate) ** 2), axis=-1)
            est = estimate[errs.argmin()]

            print(f"Estimate: ({est[0]:.2f}, {est[1]:.2f})")
            err = jnp.sum(jnp.sqrt((params.emitter - est)**2))
            print(f'# of min. points: {jnp.sum(data['cost'] == data['cost'].max())}')
            data = pd.DataFrame(data)
            (
                p9.ggplot(data, p9.aes("xs", "ys", fill="cost"))
                + p9.geom_tile()
                + p9.geom_vline(xintercept=params.emitter[0])
                + p9.geom_hline(yintercept=params.emitter[1])
                + p9.geom_vline(xintercept=est[0], linetype="dotted")
                + p9.geom_hline(yintercept=est[1], linetype="dotted")
                + p9.theme_minimal()
            ).save(f'jax_plots/dd_qpsk_{idx}_{params.seed}.png', dpi=300, width=5, height=5)

            print(f"Error: {err} m (Computed in {(end-start) * 1000:.3f}ms)")
            # break
        # break
    # jax.profiler.stop_trace()
