"""Compiled training loops for the tabular victims.

A victim's training is millions of tiny steps (sample an action, sample the next
state, update one table entry). In Python that loop is about 90% of an attack run's
time. When the environment can describe itself as a transition table
(`VictimEnvironment.tabular_model()`) and numba is installed, the tabular victims run
the same loop compiled, with identical update rules and sampling distributions.

Nothing here changes what a victim learns or what the attacker may observe; the
Python loops in each victim class remain the reference implementation and are used
whenever a fast path is not available. Set ENVPOIS_NO_JIT=1 to force them.

Random numbers: each call draws one seed from numpy's global generator, so runs stay
reproducible under np.random.seed(), but the random stream differs from the Python
loops (same distributions, different draws).
"""

import os
from typing import Optional

import numpy as np

from core.victim_environment import TabularModel

try:  # numba is optional: without it the victims use their Python loops
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only where numba is absent
    HAVE_NUMBA = False

    def njit(*_args, **_kwargs):
        return lambda fn: fn


def model_of(env) -> Optional[TabularModel]:
    """The environment's transition table, or None if the fast path cannot be used."""
    if not HAVE_NUMBA or os.environ.get("ENVPOIS_NO_JIT") == "1":
        return None
    describe = getattr(env, "tabular_model", None)
    model = describe() if describe is not None else None
    if model is None:
        return None
    return TabularModel(
        np.ascontiguousarray(model.cdf, dtype=np.float64),
        int(model.start),
        np.ascontiguousarray(model.terminal, dtype=np.bool_),
        np.ascontiguousarray(model.reward, dtype=np.float64),
        int(model.max_steps),
    )


def new_seed() -> int:
    """A seed for one compiled call, drawn from numpy's global generator."""
    return int(np.random.randint(0, 2**31 - 1))


def as_tuples(rows: np.ndarray) -> list:
    """(n, 4) int array -> list of (prev_state, prev_action, state, action) tuples."""
    return list(map(tuple, rows.tolist()))


@njit(cache=True)
def _softmax_sample(values, n):
    """Sample an index from softmax(values[:n])."""
    top = values[0]
    for i in range(1, n):
        if values[i] > top:
            top = values[i]
    total = 0.0
    for i in range(n):
        total += np.exp(values[i] - top)
    u = np.random.random() * total
    acc = 0.0
    for i in range(n):
        acc += np.exp(values[i] - top)
        if u < acc:
            return i
    return n - 1


@njit(cache=True)
def _next_state(cdf_row, n):
    """Sample the next state from one row of cumulative probabilities."""
    u = np.random.random() * cdf_row[n - 1]
    lo, hi = 0, n - 1
    while lo < hi:  # first index whose cumulative probability exceeds u
        mid = (lo + hi) // 2
        if cdf_row[mid] > u:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(cache=True)
def train_td(
    cdf,
    start,
    terminal,
    reward,
    max_steps,
    n_episodes,
    Q,
    trace,
    alpha,
    gamma,
    sarsa,
    seed,
):
    """Q-learning (sarsa=False) or SARSA (sarsa=True) with a softmax behaviour policy.

    Updates Q and trace in place. Returns per-episode returns and lengths, the
    (prev_state, prev_action, state, action) rows the victim exposes as its
    trajectory, and the (state, action) rows of the last episode.
    """
    np.random.seed(seed)
    nA, nS = cdf.shape[0], cdf.shape[1]
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=np.int64)
    rows = np.empty((n_episodes * max_steps, 4), dtype=np.int64)
    last = np.empty((max_steps, 2), dtype=np.int64)
    n_rows = 0
    n_last = 0
    for ep in range(n_episodes):
        state = start
        action = _softmax_sample(Q[state], nA) if sarsa else 0
        prev_state, prev_action = -1, -1
        total = 0.0
        n_last = 0
        t = 0
        while t < max_steps:
            if not sarsa:
                action = _softmax_sample(Q[state], nA)
            nxt = _next_state(cdf[action, state], nS)
            r = reward[nxt]
            done = terminal[nxt]
            trace[state, 1] = action
            last[n_last, 0], last[n_last, 1] = state, action
            n_last += 1
            total += r
            if sarsa:
                next_action = _softmax_sample(Q[nxt], nA)
                Q[state, action] += alpha * (
                    r + gamma * Q[nxt, next_action] - Q[state, action]
                )
                rows[n_rows, 0], rows[n_rows, 1] = state, action
                rows[n_rows, 2], rows[n_rows, 3] = nxt, next_action
                n_rows += 1
            else:
                best = Q[nxt, 0]
                for a in range(1, nA):
                    if Q[nxt, a] > best:
                        best = Q[nxt, a]
                Q[state, action] += alpha * (r + gamma * best - Q[state, action])
                if prev_state >= 0:
                    rows[n_rows, 0], rows[n_rows, 1] = prev_state, prev_action
                    rows[n_rows, 2], rows[n_rows, 3] = state, action
                    n_rows += 1
            t += 1
            if done:
                break
            prev_state, prev_action = state, action
            state = nxt
            if sarsa:
                action = next_action
        returns[ep] = total
        lengths[ep] = t
    return returns, lengths, rows[:n_rows].copy(), last[:n_last].copy()


@njit(cache=True)
def train_reinforce(
    cdf, start, terminal, reward, max_steps, n_episodes, H, trace, alpha, gamma, seed
):
    """REINFORCE with a softmax policy over action preferences H (updated in place)."""
    np.random.seed(seed)
    nA, nS = cdf.shape[0], cdf.shape[1]
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=np.int64)
    rows = np.empty((n_episodes * max_steps, 4), dtype=np.int64)
    states = np.empty(max_steps, dtype=np.int64)
    actions = np.empty(max_steps, dtype=np.int64)
    rewards = np.empty(max_steps)
    probs = np.empty(nA)
    n_rows = 0
    n = 0
    for ep in range(n_episodes):
        state = start
        total = 0.0
        n = 0
        while n < max_steps:
            action = _softmax_sample(H[state], nA)
            nxt = _next_state(cdf[action, state], nS)
            trace[state, 1] = action
            states[n], actions[n], rewards[n] = state, action, reward[nxt]
            total += reward[nxt]
            n += 1
            if terminal[nxt]:
                break
            state = nxt
        # Monte Carlo returns, then one policy-gradient step per visited (state, action)
        G = 0.0
        for i in range(n - 1, -1, -1):
            G = rewards[i] + gamma * G
            rewards[i] = G
        for i in range(n):
            s, a = states[i], actions[i]
            top = H[s, 0]
            for k in range(1, nA):
                if H[s, k] > top:
                    top = H[s, k]
            z = 0.0
            for k in range(nA):
                probs[k] = np.exp(H[s, k] - top)
                z += probs[k]
            for k in range(nA):
                grad = -probs[k] / z
                if k == a:
                    grad += 1.0
                H[s, k] += alpha * rewards[i] * grad
        for i in range(1, n):
            rows[n_rows, 0], rows[n_rows, 1] = states[i - 1], actions[i - 1]
            rows[n_rows, 2], rows[n_rows, 3] = states[i], actions[i]
            n_rows += 1
        returns[ep] = total
        lengths[ep] = n
    last = np.empty((n, 2), dtype=np.int64)
    for i in range(n):
        last[i, 0], last[i, 1] = states[i], actions[i]
    return returns, lengths, rows[:n_rows].copy(), last
