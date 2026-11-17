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


def test_push_base_follows_the_search():
    push = PushExploration((3,), prob=1.0)
    assert (push.base() == 0).all(), "undecided search gives no base"
    push.probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.4, 0.3, 0.3]])
    assert push.base(0.5).tolist() == [-1.0, 1.0, 0.0]


def test_ddpg_base_is_added_and_saved(tmp_path):
    from attack.ddpg import DDPG
    algo = DDPG(nb_states=4, nb_actions=3, max_action=1.0)
    algo.eval()
    state = np.zeros(4, dtype=np.float32)
    plain = algo.act_greedy(state)
    algo.set_base(np.array([1.0, -1.0, 0.0]))
    with_base = algo.act_greedy(state)
    assert np.allclose(with_base, np.clip(plain + [1.0, -1.0, 0.0], -1, 1), atol=1e-6)
    algo.save(str(tmp_path / "m"))
    other = DDPG(nb_states=4, nb_actions=3, max_action=1.0)
    other.load(str(tmp_path / "m"))
    other.eval()
    assert np.allclose(other.act_greedy(state), with_base, atol=1e-6)


def test_online_autoencoder_learns_observed_traces():
    from ae.ae import AEObservationEncoder
    rng = np.random.default_rng(0)
    enc = AEObservationEncoder(n_states=8, context_dim=0, model_path="online",
                               n_actions=2, online_episodes=3, online_epochs=150)
    assert enc.warmup_episodes == 3
    patterns = rng.integers(0, 2, size=(4, 8))          # four behaviours to tell apart
    for _ in range(2000):
        trace = np.column_stack([np.arange(8), patterns[rng.integers(4)]])
        enc.encode([{"behavior_trace": trace}])
    accuracy = enc.fit()
    assert enc.warmup_episodes == 0, "frozen after the fit"
    assert accuracy > 0.9
    z = [enc.encode([{"behavior_trace": np.column_stack([np.arange(8), p])}]) for p in patterns]
    assert not np.allclose(z[0], z[1]), "different behaviours get different embeddings"


def test_config_helpers():
    assert parse_kv("n_states:12,gamma:0.5,name:x") == {"n_states": 12, "gamma": 0.5, "name": "x"}
    cfg = resolve_config({"env": "ring", "env_args": "n_states:12", "not_a_setting": 1})
    assert cfg["env_kwargs"] == {"n_states": 12} and "not_a_setting" not in cfg
    assert resolve_config({"grid_size": 5})["env_kwargs"]["grid_dimensions"] == (5, 5)


def test_run_naming_round_trip():
    from naming import parse_run_name, run_name, run_path
    cases = [
        ({"env": "grid3d", "grid_size": 5, "victim_algo": "sarsa", "seed": 0, "experiment": "sweep-3x3x3"},
         "sweep-3x3x3/grid3d-5x5__sarsa__default__seed0"),
        ({"env": "ring", "env_args": "n_states:12", "victim_algo": "qlearning", "seed": 2},
         "adhoc/ring-n12__qlearning__default__seed2"),
        ({"env": "lunarlander", "env_args": "initial_random:1000", "victim_algo": "dqn", "seed": 1,
          "config_name": "TD3 on"}, "adhoc/lunarlander-kick1000__dqn__td3-on__seed1"),
        ({"env": "grid3d", "victim_algo": "qlearning", "seed": 7}, "adhoc/grid3d-4x4__qlearning__default__seed7"),
    ]
    for settings, expected in cases:
        cfg = resolve_config(settings)
        assert run_path(cfg) == expected
        parts = parse_run_name(run_name(cfg))
        assert parts["env"] == cfg["env"] and parts["seed"] == cfg["seed"]
