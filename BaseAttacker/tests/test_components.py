"""Unit tests for the pieces with real logic in them."""
import numpy as np

from attack.push_exploration import PushExploration
from builders import parse_kv, resolve_config
from envs.grid3d import Grid3D
from utils.metrics import accuracy_from_behavior_trace, greedy_path, target_action_accuracy


def test_grid_transitions_are_probabilities():
    for shape in [(4, 4), (3, 5), (6, 6)]:
        env = Grid3D(grid_dimensions=shape)
        env.apply_perturbation(np.random.default_rng(0).uniform(-1, 1, env.nS))
        assert np.allclose(env.T.sum(axis=2), 1.0)
        assert (env.T >= 0).all()


def test_grid_natural_route_is_straight_down():
    env = Grid3D()
    down = np.full(env.nS, 2)                      # SOUTH everywhere
    assert greedy_path(env.T, down, 0, 12) == [0, 4, 8, 12]


def test_grid_target_is_the_perimeter():
    env = Grid3D()
    target = env.make_target_policy()
    actions = target.argmax(axis=1)
    assert greedy_path(env.INIT_T, actions, 0, 12) == [0, 1, 2, 3, 7, 11, 15, 14, 13, 12]
    assert env.oracle_route_match(actions, actions) == 1.0
    assert env.oracle_route_match(actions, np.full(env.nS, 2)) == 0.0


def test_accuracy_measures_agree_on_a_perfect_victim():
    env = Grid3D()
    target = env.make_target_policy()
    assert target_action_accuracy(target, target) == 1.0
    trace = np.column_stack([np.arange(env.nS), target.argmax(axis=1)])
    assert accuracy_from_behavior_trace(trace, target) == 1.0


def test_push_exploration_learns_the_rewarded_direction():
    np.random.seed(0)
    push = PushExploration((6,), prob=1.0, adaptive=True, batch=8, elite=2)
    for _ in range(400):
        assert push.start_episode(warmup=False)
        action = push.action()
        assert action.shape == (6,) and np.abs(action).max() <= 1.0
        push.end_episode(float(push._current[0] == 2))     # reward "up" on dimension 0
    assert push.probs[0, 2] > 0.8
    assert np.allclose(push.probs.sum(axis=1), 1.0)


def test_config_helpers():
    assert parse_kv("n_states:12,gamma:0.5,name:x") == {"n_states": 12, "gamma": 0.5, "name": "x"}
    cfg = resolve_config({"env": "ring", "env_args": "n_states:12", "not_a_setting": 1})
    assert cfg["env_kwargs"] == {"n_states": 12} and "not_a_setting" not in cfg
    assert resolve_config({"grid_size": 5})["env_kwargs"]["grid_dimensions"] == (5, 5)
