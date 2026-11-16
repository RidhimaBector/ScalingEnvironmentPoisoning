"""Contract tests: every registered environment and victim algorithm must pass these.

If you add your own environment or victim (see README), register it in registry.py
and run `pytest tests/test_interfaces.py`. Passing here means the attack pipeline can
drive it without knowing anything else about it.
"""
import numpy as np
import pytest

from registry import ENV_MAP, VICTIM_ALGO_MAP

TABULAR_VICTIMS = ["qlearning", "sarsa", "reinforce"]
NEURAL_VICTIMS = ["dqn", "vpg", "a2c"]


def make_env(name):
    try:
        return ENV_MAP[name]()
    except ImportError as exc:      # optional dependency (Box2D for the lander)
        pytest.skip(f"{name}: {exc}")


def is_tabular(env):
    return hasattr(env.observation_space, "n")


@pytest.mark.parametrize("name", sorted(ENV_MAP))
def test_environment_contract(name):
    env = make_env(name)
    space = env.perturbation_space

    # dynamics vector
    start = np.asarray(env.get_dynamics(), dtype=float)
    assert start.ndim == 1 and start.shape[0] == env.dynamics_dim

    # a perturbation changes the dynamics; reset_dynamics() undoes it
    env.apply_perturbation(np.asarray(space.high, dtype=float))
    assert not np.allclose(env.get_dynamics(), start), "apply_perturbation had no effect"
    env.reset_dynamics()
    assert np.allclose(env.get_dynamics(), start), "reset_dynamics did not restore the start"

    bounds = env.dynamics_bounds()
    if bounds is not None:
        assert bounds[0] <= start.min() and start.max() <= bounds[1]

    # exactly one way of stating the target
    target, recorded = env.make_target_policy(), env.target_trajectories()
    assert (target is None) != (recorded is None), "define a target policy OR target trajectories"
    if target is not None:
        assert is_tabular(env), "a target policy table needs discrete states"
        assert target.shape == (env.observation_space.n, env.action_space.n)
        assert set(np.unique(target)) <= {0.0, 1.0} and target.sum() > 0
    else:
        prev_state, prev_action, state, action = recorded[0]
        assert np.asarray(state).shape == env.observation_space.shape
        assert 0 <= int(action) < env.action_space.n

    # gymnasium episode API
    env.seed(0)
    obs, info = env.reset()
    assert isinstance(info, dict)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert np.isscalar(reward) and isinstance(info, dict)


@pytest.mark.parametrize("env_name", sorted(ENV_MAP))
@pytest.mark.parametrize("algo_name", sorted(VICTIM_ALGO_MAP))
def test_victim_contract(env_name, algo_name):
    env = make_env(env_name)
    if is_tabular(env) != (algo_name in TABULAR_VICTIMS):
        pytest.skip("tabular victims need discrete states; neural victims need vector states")
    env.max_steps = 30                              # keep the test fast
    algo_class = VICTIM_ALGO_MAP[algo_name]
    if hasattr(algo_class, "from_env"):
        algo = algo_class.from_env(env)
    else:
        algo = algo_class(env_nS=env.observation_space.n, env_nA=env.action_space.n)

    np.random.seed(0)
    stats = algo.train(env, 3)
    assert len(stats["episode_rewards"]) == 3

    transitions = algo.get_trajectories()
    assert transitions, "the victim must expose (prev_state, prev_action, state, action) tuples"
    assert len(transitions[0]) == 4
    assert algo.get_trajectories() == [], "get_trajectories() must clear its buffer"

    if is_tabular(env):
        trace = algo.get_behavior_trace()
        assert trace.shape == (env.observation_space.n, 2)
        assert algo.get_policy_matrix().shape == (env.observation_space.n, env.action_space.n)
    else:
        states = np.stack([t[2] for t in transitions[:5]])
        assert algo.greedy_actions_for(states).shape == (len(states),)

    algo.reset()
