import enum
from dataclasses import dataclass, field
from typing import Optional, TypeAlias, Callable, Any

from jaxtyping import Float
import numpy as np
import pint
import plotnine as p9

SEED = 42

Q = pint.Quantity

ureg = pint.UnitRegistry(force_ndarray=True)

PROPOGATION_SPEED = 3e8 * ureg.meter_per_second
# TODO: convert to proper unit
NOMINAL_CARRIER_FREQUENCY = 100 * ureg.MHz
DEFAULT_RECEIVER_SPEED = 300 * ureg.meter_per_second

NUM_SAMPLES_PER_SECOND: int = 100

@dataclass(repr=True)
class Params:
    seed: int
    rng: np.random.Generator = field(init=False)

    num_receivers: int
    num_timesteps: int
    num_samples_per_interval: int

    noise_stddev: float

    receivers_p_x: Q[Float[np.ndarray, "receivers timesteps"]]
    receivers_p_y: Q[Float[np.ndarray, "receivers timesteps"]]
    receivers_v_x: Q[Float[np.ndarray, "receivers timesteps"]]
    receivers_v_y: Q[Float[np.ndarray, "receivers timesteps"]]
    
    emitter_x: Q[int | float] = field(init=False)
    emitter_y: Q[int | float] = field(init=False)

    timesteps: Q[Float[np.ndarray, "timesteps"]] = field(init=False)

    transmitted_freq_shifts: Q[Float[np.ndarray, "receivers"]] = field(init=False)
    channel_attenuation: Q[float | int] = field(init=False)
    channel_phase: Q[float | int] = field(init=False)

    def __post_init__(self):
        # check if we have the correct number
        req_shape = (self.num_receivers, self.num_timesteps)
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
        vx, vy = self.receivers_v_x[0], self.receivers_v_y[0]
        px, py = self.receivers_p_x[0], self.receivers_p_y[0]
        zero = 0.0 * ureg.second

        timesteps_x = np.where(vx != 0, (px / vx).to(ureg.second), zero)
        timesteps_y = np.where(vy != 0, (py / vy).to(ureg.second), zero)

        self.timesteps = np.maximum(timesteps_x, timesteps_y)

class DefaultPosition(enum.StrEnum):
    PosA = 'A'
    PosB = 'B'
    PosC = 'C'
    PosD = 'D'


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
                ]) * km,
                receivers_p_y=np.stack([
                    np.full(T, 0.0, dtype=np.float64),
                    np.full(T, 10.0, dtype=np.float64)
                ]) * km,
                receivers_v_x=np.full((L, T), DEFAULT_RECEIVER_SPEED, dtype=np.float64) * mps,
                receivers_v_y=np.zeros((L, T)) * mps
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
                ]) * km,
                receivers_p_y=np.stack([
                    np.full(T, 0.0, dtype=np.float64),
                    np.full(T, 10.0, dtype=np.float64),
                    np.full(T, 0.2, dtype=np.float64)
                ]) * km,
                receivers_v_x=np.full((L, T), DEFAULT_RECEIVER_SPEED, dtype=np.float64) * mps,
                receivers_v_y=np.zeros((L, T)) * mps
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
                ]) * km,
                receivers_p_y=np.stack([
                    np.full(T, 0.0),
                    np.arange(1, 11, dtype=np.float64)
                ]) * km,
                receivers_v_x=np.stack([
                    np.full(T, DEFAULT_RECEIVER_SPEED, dtype=np.float64),
                    np.zeros(T)
                ]) * mps,
                receivers_v_y=np.stack([
                    np.zeros(T),
                    np.full(T, DEFAULT_RECEIVER_SPEED, dtype=np.float64)
                ]) * mps
            )

        case DefaultPosition.PosD:
            L, T = 2, 10
            return Params(
                seed=SEED,
                num_receivers=L,
                num_timesteps=T,
                num_samples_per_interval=N,
                noise_stddev=1.0,
                receivers_p_x=np.tile(np.arange(1, 11, dtype=np.float64), (L,1)) * km,
                receivers_p_y=np.stack([
                    np.full(T, 0.0, dtype=np.float64),
                    np.full(T, -0.5, dtype=np.float64)
                ]) * km,
                receivers_v_x=np.full((L, T), DEFAULT_RECEIVER_SPEED, dtype=np.float64) * mps,
                receivers_v_y=np.zeros((L, T)) * mps
            )
def calculate_mu(
        p0_x: Q[float],
        p0_y: Q[float],
        p_lk_x: Q[Float[np.ndarray, "l k"]],
        p_lk_y: Q[Float[np.ndarray, "l k"]],
        v_lk_x: Q[Float[np.ndarray, "l k"]],
        v_lk_y: Q[Float[np.ndarray, "l k"]],
        c: float
) -> Q[Float[np.ndarray, "l k"]]:
    p_diff_x = p0_x - p_lk_x
    p_diff_y = p0_y - p_lk_y
    
    numerator = v_lk_x * p_diff_x + v_lk_y * p_diff_y
    denominator = np.sqrt(p_diff_x ** 2 + p_diff_y ** 2)
    
    return numerator / (c * denominator)
    
def simulate_signal(params: Params, generate_signal: Callable[[int, int], Q[Float[np.ndarray, "timesteps samples"]]]) -> Q[Float[np.ndarray, ""]]:
    l = params.num_receivers
    k = params.num_timesteps
    N = params.num_samples_per_interval

    # TODO: find a way to simulate this
    b: Q[Float[np.ndarray, "l k"]] = np.ones((l, k)) * ureg.dimensionless
    # TODO: find a way to simulate this
    s: Q[Float[np.ndarray, "k N"]] = generate_signal(k, N)

    # TODO: find normal std deviation
    w: Q[Float[np.ndarray, "l k N"]] = params.rng.normal(0, params.noise_stddev, (l, k, N)) * ureg.dimensionless

    mu: Q[Float[np.ndarray, "l k"]] = calculate_mu(
        params.emitter_x, params.emitter_y,
        params.receivers_p_x, params.receivers_p_y,
        params.receivers_v_x, params.receivers_v_y,
        c=PROPOGATION_SPEED
    )
    # TODO: find interval time
    # is it just 100 samples / 10 ksamples/s?
    T_s = 100 / 10000
    exps = np.arange(0, N * T_s, T_s) * ureg.s

    # Here, instead of taking the diagonal matrix, we just mutliply them elementwise since it reduces the # of ops
    A: Q[Float[np.ndarray, "l k N"]] = np.exp(1j * 2 * np.pi * NOMINAL_CARRIER_FREQUENCY * np.einsum('lk,n->lkn', mu, exps))
    C: Q[Float[np.ndarray, "k N"]] = np.exp(1j * 2 * np.pi * np.einsum('k,n->kn', params.transmitted_freq_shifts, exps))

    return np.einsum('lk,lkn->lkn', b, A * C * s)

def estimate_direct_position(params: Params, signal: Q[Float[np.ndarray, "receiver timestep interval"]], p_min: Q[float], p_max: Q[float], p_step: Q[float]) -> tuple[Q[float], Q[float]]:
    p_lk_x = params.receivers_p_x
    p_lk_y = params.receivers_p_y
    v_lk_x = params.receivers_v_x
    v_lk_y = params.receivers_v_y

    c = PROPOGATION_SPEED

    N = params.num_samples_per_interval

    max_sum = -np.inf
    p_best = (-np.inf * ureg.m, -np.inf * ureg.m)

    # TODO: ensure that units are correct
    for p_x in np.arange(p_min.m, p_max.m, p_step.m) * p_min.u:
        for p_y in np.arange(p_min.m, p_max.m, p_step.m) * p_min.u:
            mu = calculate_mu(p_x, p_y, p_lk_x, p_lk_y, v_lk_x, v_lk_y, c)

            # TODO: calculate this properly
            T_s = 100 / 10000
            exps = np.arange(0, N * T_s, T_s) * ureg.s
            # shape: [l, k, n]
            A = np.exp(1j * 2 * np.pi * NOMINAL_CARRIER_FREQUENCY * np.einsum('lk,n->lkn', mu, exps))

            V: Q[Float[np.ndarray, "l k n"]] = A * signal
            V: Q[Float[np.ndarray, "k l n"]] = V.transpose(1, 0, 2)
            Q_k: Q[Float[np.ndarray, "k l l"]] = V @ V.conj().transpose(0, 2, 1)

            eigen_val = np.sum(np.linalg.eigvalsh(Q_k)[:, 0])
            print((p_x, p_y), eigen_val, max_sum)

            if eigen_val > max_sum:
                print(eigen_val, max_sum)
                print(p_best, (p_x, p_y))
                max_sum = eigen_val
                p_best = (p_x, p_y)

    return p_best

def sin_signal(k: float, N: float):
    k_vals = np.arange(k)
    n_vals = np.arange(N)
    return np.sin(2 * np.pi * np.outer(k_vals, n_vals) / (k * N) + 0.5)

if __name__ == '__main__':
    for pos in [DefaultPosition.PosA, DefaultPosition.PosB, DefaultPosition.PosC, DefaultPosition.PosD]:
        params = init_position(pos)
        print(params.emitter_x, params.emitter_y)
        signal = simulate_signal(params, sin_signal)
        print(estimate_direct_position(params, signal, 0 * ureg.km, 10 * ureg.km, 0.1 * ureg.km))
