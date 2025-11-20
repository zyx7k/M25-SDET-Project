import traceback
import sys
import enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Any

from jaxtyping import Float
import numpy as np
import pandas as pd
import pint
import plotnine as p9

SEED = 42

Q = pint.Quantity

ureg = pint.UnitRegistry(force_ndarray=True)

PROPOGATION_SPEED: Q[float] = 3e8 * ureg.meter_per_second
# TODO: convert to proper unit
NOMINAL_CARRIER_FREQUENCY: Q[int] = 100 * ureg.MHz
DEFAULT_RECEIVER_SPEED: Q[float] = 300 * ureg.meter_per_second

NUM_SAMPLES_PER_SECOND: int = 100

@dataclass(repr=True)
class Params:
    seed: int
    rng: np.random.Generator = field(init=False)

    num_receivers: int
    num_timesteps: int
    num_samples_per_interval: int

    noise_stddev: float

    receivers_p_x: Q[Float[np.ndarray, "timesteps receivers"]]
    receivers_p_y: Q[Float[np.ndarray, "timesteps receivers"]]
    receivers_v_x: Q[Float[np.ndarray, "timesteps receivers"]]
    receivers_v_y: Q[Float[np.ndarray, "timesteps receivers"]]

    emitter_x: Q[int | float] = field(init=False)
    emitter_y: Q[int | float] = field(init=False)

    timesteps: Q[Float[np.ndarray, "timesteps"]] = field(init=False)

    transmitted_freq_shifts: Q[Float[np.ndarray, "receivers"]] = field(init=False)
    channel_attenuation: Q[float | int] = field(init=False)
    channel_phase: Q[float | int] = field(init=False)

    def __post_init__(self):
        # check if we have the correct number
        req_shape = (self.num_timesteps, self.num_receivers)
        assert self.receivers_p_x.shape == req_shape
        assert self.receivers_p_y.shape == req_shape
        assert self.receivers_v_x.shape == req_shape
        assert self.receivers_v_y.shape == req_shape

        self.rng = np.random.default_rng(self.seed)

        # The emitter’s position is chosen at random within a square area of 10 x 10 [Km x Km].
        self.emitter_x = self.rng.uniform(0, 10_000) * ureg.meter
        self.emitter_y = self.rng.uniform(0, 10_000) * ureg.meter

        # The unknown transmitted frequency shifts, {\mathcal{v}_k}, are selected at random
        # from the interval [-100, 100] Hz
        self.transmitted_freq_shifts = self.rng.uniform(-100, 100, self.num_timesteps) * ureg.Hz
        # The channel attenuation is selected at random from a normal distribution with
        # mean one and standard deviation 0.1
        self.channel_attenuation = self.rng.normal(1, 0.1) * ureg.Hz
        # and the channel phase is selected at random from a
        # uniform distribution over [-pi, pi]
        self.channel_phase = self.rng.uniform(-np.pi, np.pi) * ureg.Hz

        # we assume that the velocity vector remains constant between each interception interval
        # we also assume that the interception times are the same for all receivers
        # therefore, we can calculate the time simply by taking the max time along x and y
        # (just incase any direction is 0)
        vx, vy = self.receivers_v_x[:, 0], self.receivers_v_y[:, 0]
        px, py = self.receivers_p_x[:, 0], self.receivers_p_y[:, 0]
        zero = 0.0 * ureg.second

        timesteps_x: Q[Float[np.ndarray, "timesteps receivers"]] = np.where(vx != 0, (px / vx).to(ureg.second), zero)
        timesteps_y: Q[Float[np.ndarray, "timesteps receivers"]] = np.where(vy != 0, (py / vy).to(ureg.second), zero)

        self.timesteps = np.maximum(timesteps_x, timesteps_y)

class DefaultPosition(enum.StrEnum):
    PosA = 'A'
    PosB = 'B'
    PosC = 'C'
    PosD = 'D'
    PosTest = 'Test'


def init_position(position: DefaultPosition) -> Params:
    km = ureg.kilometer
    mps = ureg.meter_per_second

    N = 100

    match position:
        case DefaultPosition.PosA:
            L, T = 2, 10
            return Params(
                seed=SEED,
                num_receivers=L,
                num_timesteps=T,
                num_samples_per_interval=N,
                noise_stddev=1.0,
                receivers_p_x=np.stack([
                    np.arange(1, 11, dtype=np.float64),
                    np.arange(10, 0, -1, dtype=np.float64)
                ]).T * km,
                receivers_p_y=np.stack([
                    np.full(T, 0.0, dtype=np.float64),
                    np.full(T, 10.0, dtype=np.float64)
                ]).T * km,
                receivers_v_x=np.full((T, L), DEFAULT_RECEIVER_SPEED, dtype=np.float64) * mps,
                receivers_v_y=np.zeros((T, L)) * mps
            )

        case DefaultPosition.PosB:
            L, T = 3, 10
            return Params(
                seed=SEED,
                num_receivers=L,
                num_timesteps=T,
                num_samples_per_interval=N,
                noise_stddev=1.0,
                receivers_p_x=np.stack([
                    np.arange(1, 11, dtype=np.float64),
                    np.arange(10, 0, -1, dtype=np.float64),
                    np.arange(1, 11, dtype=np.float64)
                ]).T * km,
                receivers_p_y=np.stack([
                    np.full(T, 0.0, dtype=np.float64),
                    np.full(T, 10.0, dtype=np.float64),
                    np.full(T, 0.2, dtype=np.float64)
                ]).T * km,
                receivers_v_x=np.full((T, L), DEFAULT_RECEIVER_SPEED, dtype=np.float64) * mps,
                receivers_v_y=np.zeros((T, L)) * mps
            )

        case DefaultPosition.PosC:
            L, T = 2, 10
            return Params(
                seed=SEED,
                num_receivers=L,
                num_timesteps=T,
                num_samples_per_interval=N,
                noise_stddev=1.0,
                receivers_p_x=np.stack([
                    np.arange(1, 11, dtype=np.float64),
                    np.full(T, 10.0)
                ]).T * km,
                receivers_p_y=np.stack([
                    np.full(T, 0.0),
                    np.arange(1, 11, dtype=np.float64)
                ]).T * km,
                receivers_v_x=np.stack([
                    np.full(T, DEFAULT_RECEIVER_SPEED, dtype=np.float64),
                    np.zeros(T)
                ]).T * mps,
                receivers_v_y=np.stack([
                    np.zeros(T),
                    np.full(T, DEFAULT_RECEIVER_SPEED, dtype=np.float64)
                ]).T * mps
            )

        case DefaultPosition.PosD:
            L, T = 2, 10
            return Params(
                seed=SEED,
                num_receivers=L,
                num_timesteps=T,
                num_samples_per_interval=N,
                noise_stddev=1.0,
                receivers_p_x=np.tile(np.arange(1, 11, dtype=np.float64), (L,1)).T * km,
                receivers_p_y=np.stack([
                    np.full(T, 0.0, dtype=np.float64),
                    np.full(T, -0.5, dtype=np.float64)
                ]).T * km,
                receivers_v_x=np.full((T, L), DEFAULT_RECEIVER_SPEED, dtype=np.float64) * mps,
                receivers_v_y=np.zeros((T, L)) * mps
            )
        case DefaultPosition.PosTest:
            L, T = 2, 4       # two receivers, one timestep
            N = 128           # moderately sized interval

            m = ureg.m
            s = ureg.s
            
            # Receiver positions
            receivers_p_x = np.array([[0.0, 1000.0], [300.0, 1300.0], [600.0, 1600.0], [900.0, 1900.0]]) * m
            receivers_p_y = np.zeros((T, L)) * m
            
            # Non-zero velocities: both move east at 300 m/s
            receivers_v_x = np.full((T, L), 300) * m/s
            receivers_v_y = np.zeros((T, L)) * m/s
            
            params = Params(
                seed=SEED,
                num_receivers=L,
                num_timesteps=T,
                num_samples_per_interval=N,
                noise_stddev=0.0,  # no noise for a clean test
                receivers_p_x=receivers_p_x,
                receivers_p_y=receivers_p_y,
                receivers_v_x=receivers_v_x,
                receivers_v_y=receivers_v_y,
            )
            
            # Fixed emitter position (known)
            params.emitter_x = 400 * m
            params.emitter_y = 300 * m
            
            # No transmitter frequency shifts for the test
            params.transmitted_freq_shifts = np.zeros(T) * ureg.Hz
            
            return params

def calculate_mu(
        p0_x: Q[float],
        p0_y: Q[float],
        p_lk_x: Q[Float[np.ndarray, "k l"]],
        p_lk_y: Q[Float[np.ndarray, "k l"]],
        v_lk_x: Q[Float[np.ndarray, "k l"]],
        v_lk_y: Q[Float[np.ndarray, "k l"]],
        c: Q[float]
) -> Q[Float[np.ndarray, "k l"]]:
    p_diff_x = p0_x - p_lk_x
    p_diff_y = p0_y - p_lk_y

    numerator = v_lk_x * p_diff_x + v_lk_y * p_diff_y
    denominator = np.sqrt(p_diff_x ** 2 + p_diff_y ** 2)
    denominator[denominator == 0] = 1e-9 * ureg.m

    return numerator / (c * denominator)

def calculate_mu_vectorized(p0_x, p0_y, p_lk_x, p_lk_y, v_lk_x, v_lk_y, c):
    """
    p0_x: (G,) array of candidate x points
    p0_y: (G,) array of candidate y points
    p_lk_x, p_lk_y: (k,l)
    v_lk_x, v_lk_y: (k,l)
    c: scalar
    Returns: (G, k, l)
    """

    # Expand G dimension
    p0_x = p0_x[:, None, None]
    p0_y = p0_y[:, None, None]

    p_diff_x = p0_x - p_lk_x[None,:,:]
    p_diff_y = p0_y - p_lk_y[None,:,:]

    numerator = v_lk_x[None,:,:] * p_diff_x + v_lk_y[None,:,:] * p_diff_y
    denominator = np.sqrt(p_diff_x**2 + p_diff_y**2)
    denominator = np.maximum(denominator, 1e-9 * ureg.m)

    return numerator / (c * denominator)

def simulate_signal(params: Params, generate_signal: Callable[[int, int], Float[np.ndarray, "timesteps interval"]]) -> Q[Float[np.ndarray, "timesteps receivers interval"]]:
    l = params.num_receivers
    k = params.num_timesteps
    N = params.num_samples_per_interval

    # TODO: find a way to simulate this
    b: Q[Float[np.ndarray, "k l"]] = np.ones((k, l)) * ureg.dimensionless
    # TODO: find a way to simulate this
    s: Float[np.ndarray, "k N"] = generate_signal(k, N)

    # TODO: find normal std deviation
    w: Q[Float[np.ndarray, "k l N"]] = params.rng.normal(0, params.noise_stddev, (k, l, N)) * ureg.dimensionless

    mu: Q[Float[np.ndarray, "k l"]] = calculate_mu(
        params.emitter_x, params.emitter_y,
        params.receivers_p_x, params.receivers_p_y,
        params.receivers_v_x, params.receivers_v_y,
        c=PROPOGATION_SPEED
    )
    # TODO: find interval time
    # is it just 100 samples / 10 ksamples/s?
    T_s = 100 / 10000 * ureg.s
    exps: Q[Float[np.ndarray, "N"]] = np.arange(N) * T_s

    # Here, instead of taking the diagonal matrix, we just mutliply them elementwise since it reduces the # of ops
    A: Q[Float[np.ndarray, "k l N"]] = np.exp(1j * 2 * np.pi * NOMINAL_CARRIER_FREQUENCY * np.einsum('kl,n->kln', mu, exps))
    C: Q[Float[np.ndarray, "k N"]] = np.exp(1j * 2 * np.pi * np.einsum('k,n->kn', params.transmitted_freq_shifts, exps))

    print(f"{l=}, {k=}, {N=}")
    print(f"{exps.shape=}, {b.shape=}, {A.shape=}, {C.shape=}, {s.shape=}")

    # TODO: rewrite with einsum
    signal: Q[Float[np.ndarray, "k l N"]] = b[:, :, None] * A * C[:, None, :] * s[:, None, :]
    print(signal.shape)
    return signal + w

def estimate_direct_position(params: Params, signal: Q[Float[np.ndarray, "timestep receiver interval"]], p_min: Q[float], p_max: Q[float], p_step: Q[float], generate_prior_signal: Optional[ Callable[[int, int], Float[np.ndarray, "timesteps interval"]]] = None) -> tuple[Q[float], Q[float]]:
    p_lk_x = params.receivers_p_x
    p_lk_y = params.receivers_p_y
    v_lk_x = params.receivers_v_x
    v_lk_y = params.receivers_v_y

    c = PROPOGATION_SPEED

    N = params.num_samples_per_interval

    prior_signal = None
    if generate_prior_signal is not None:
        k = params.num_timesteps
        prior_signal = generate_prior_signal(k, N)

    max_sum = -np.inf
    p_best = (-np.inf * ureg.m, -np.inf * ureg.m)

    # TODO: ensure that units are correct
    xs = np.arange(p_min.to('m').magnitude,
                   p_max.to('m').magnitude,
                   p_step.to('m').magnitude)
    ys = np.arange(p_min.to('m').magnitude,
                   p_max.to('m').magnitude,
                   p_step.to('m').magnitude)

    xs, ys = np.meshgrid(xs, ys)
    xs = xs.ravel() * ureg.m
    ys = ys.ravel() * ureg.m

    mus = calculate_mu_vectorized(xs, ys, p_lk_x, p_lk_y, v_lk_x, v_lk_y, c)

    data = {
        "xs": [],
        "ys": [],
        "cost": []
    }
    for p_x, p_y, mu in zip(xs, ys, mus):
            # mu = calculate_mu(p_x, p_y, p_lk_x, p_lk_y, v_lk_x, v_lk_y, c)

            # TODO: calculate this properly
            T_s = 100 / 10000
            exps = np.arange(0, N * T_s, T_s) * ureg.s
            # shape: [k, l, n]
            A: Q[Float[np.ndarray, "k l n"]] = np.exp(1j * 2 * np.pi * NOMINAL_CARRIER_FREQUENCY * np.einsum('kl,n->kln', mu, exps))

            V: Q[Float[np.ndarray, "k l n"]] = np.conjugate(A) * signal

            if generate_prior_signal is None:
                # we run the algorithm for unknown signal
                Q_k = np.einsum('kln,kmn->klm', V, np.conjugate(V))

                cost = np.sum(np.linalg.eigvalsh(Q_k)[:, -1]).m
                # print((p_x, p_y), eigen_val, max_sum)
            else:
                assert prior_signal is not None
                # we run the algorithm for known signal
                B = np.einsum('kln,kn->kln', V, prior_signal)
                G = np.einsum('kli,klj->kij', B.conj(), B)
                # 3. Extract Diagonals (Summing for alpha) [Shape: K, N]
                alpha = np.zeros_like(prior_signal, dtype=complex)
                # We calculate the sums of the diagonals (lags 0 to N-1)
                for m in range(N):
                    # G is dimensionless anyways
                    alpha[:, m] = np.trace(G.m, offset=m, axis1=1, axis2=2)
                beta = alpha
                beta[:, 1:] *= 2

                # perform fft
                fft_out = np.fft.fft(beta.conj(), n=N)
                cost = np.max(np.real(fft_out), axis=1)
                cost = np.sum(cost)

            data['xs'].append(p_x.to('m').m)
            data['ys'].append(p_y.to('m').m)
            data['cost'].append(cost)

            if cost > max_sum:
                print(cost, max_sum)
                print(p_best, (p_x, p_y))
                max_sum = cost
                p_best = (p_x, p_y)

    data = {key: np.array(data[key]) for key in data}
    return p_best, data

def sin_signal(k: int, N: int):
    k_vals = np.arange(k)
    n_vals = np.arange(N)
    return np.sin(2 * np.pi * np.outer(k_vals, n_vals) / (k * N) + 0.5)

if __name__ == '__main__':
    for idx, pos in enumerate([DefaultPosition.PosA, DefaultPosition.PosB, DefaultPosition.PosC, DefaultPosition.PosD]):
        params = init_position(pos)
        print(params.emitter_x, params.emitter_y)
        signal = simulate_signal(params, sin_signal)
        estimate, data = estimate_direct_position(params, signal, 0 * ureg.km, 10 * ureg.km, 0.1 * ureg.km, generate_prior_signal=sin_signal)
        print(estimate)
        print(params.emitter_x, params.emitter_y)

        data = pd.DataFrame(data)
        (
            p9.ggplot(data, p9.aes("xs", "ys", fill="cost"))
            + p9.geom_tile()
            + p9.geom_vline(xintercept=params.emitter_x.to('m').m)
            + p9.geom_hline(yintercept=params.emitter_y.to('m').m)
            + p9.theme_minimal()
        ).save(f'plots/plot_{idx}.png', dpi=300, width=5, height=5)
