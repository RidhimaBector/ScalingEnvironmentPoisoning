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


def test_push_exploration_widens_when_stalled_and_resets_on_progress():
    push = PushExploration((6,), prob=1.0, adaptive=True, batch=4, elite=2, patience=2)
    floor = push.floor

    def batch(reward):
        for _ in range(4):
            push.start_episode(warmup=False)
            push.end_episode(reward)

    batch(1.0)                       # first batch sets the best elite reward
    assert push.floor_scale == 1.0
    batch(1.0), batch(1.0)           # two batches without a rise
    assert push.floor_scale == 2.0
    assert push.probs.min() >= 2 * floor / (1 + 6 * floor) - 1e-9
    batch(1.0), batch(1.0)
    assert push.floor_scale == 4.0
    batch(1.0), batch(1.0)
    assert push.floor_scale == 4.0   # capped
    batch(2.0)                       # progress
    assert push.floor_scale == 1.0
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
    import torch
    from ae.ae import AEObservationEncoder
    torch.manual_seed(0)
    np.random.seed(0)
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


def _victims_after_training(env_fn, algo, monkeypatch, compiled, victims=40, episodes=300):
    """Mean length of each victim's last 100 episodes, for the Python or compiled loop."""
    from registry import VICTIM_ALGO_MAP
    monkeypatch.setenv("ENVPOIS_NO_JIT", "0" if compiled else "1")
    out = []
    for seed in range(victims):
        np.random.seed(seed)
        env = env_fn()
        victim = VICTIM_ALGO_MAP[algo](env.nS, env.nA)
        stats = victim.train(env, episodes)
        assert len(victim.get_trajectories()) > 0
        out.append(np.mean(stats["episode_lengths"][-100:]))
    return np.array(out)


def test_compiled_victims_learn_like_the_python_loops(monkeypatch):
    """The compiled training loops are a speed-up only: same learning, statistically."""
    import pytest
    from envs.ring import SlipperyRing
    from victim import fast_tabular
    if not fast_tabular.HAVE_NUMBA:
        pytest.skip("numba not installed: victims use the Python loops")
    for env_fn in (Grid3D, lambda: SlipperyRing(8)):
        assert fast_tabular.model_of(env_fn()) is not None
        for algo in ("qlearning", "sarsa"):
            slow = _victims_after_training(env_fn, algo, monkeypatch, compiled=False)
            fast = _victims_after_training(env_fn, algo, monkeypatch, compiled=True)
            spread = np.sqrt(slow.var() / len(slow) + fast.var() / len(fast))
            assert abs(slow.mean() - fast.mean()) < 4 * spread + 0.05, (algo, slow.mean(), fast.mean())
    # REINFORCE: too slow in Python to compare here; check the compiled loop runs
    fast = _victims_after_training(Grid3D, "reinforce", monkeypatch, compiled=True, victims=2, episodes=20)
    assert np.isfinite(fast).all()


def test_tabular_model_matches_step():
    """The transition table an environment reports is the one its step() samples from."""
    from envs.ring import SlipperyRing
    for env in (Grid3D(), SlipperyRing(8)):
        env.apply_perturbation(np.random.default_rng(1).uniform(-1, 1, env.perturbation_space.shape))
        model = env.tabular_model()
        probs = np.diff(model.cdf, axis=2, prepend=0.0)
        np.random.seed(0)
        for action in range(env.nA):
            counts = np.zeros(env.nS)
            for _ in range(4000):
                env.reset()
                nxt = env.step(action)[0]
                counts[nxt] += 1
            expected = probs[action, model.start] / probs[action, model.start].sum()
            assert np.abs(counts / 4000 - expected).max() < 0.04


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


def test_route_prefix_counts_target_states_until_first_deviation():
    from utils.metrics import route_prefix_from_trajectories

    # 4 states in a line, target: action 1 in states 0, 1, 2 (state 3 is the end)
    target = np.zeros((4, 2))
    target[[0, 1, 2], 1] = 1
    follows = [(0, 1, 1, 1), (1, 1, 2, 1)] * 5
    assert route_prefix_from_trajectories(follows, target) == 1.0
    # the target action usually fails in state 1: the walk ends there (states 0, 1 count)
    stuck = [(0, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 2, 1)] * 5
    assert route_prefix_from_trajectories(stuck, target) == 1.0  # ignored by default
    strict = route_prefix_from_trajectories(stuck, target, stop_when_stuck=True)
    assert abs(strict - 2 / 3) < 1e-9
    # an occasional failed move does not end it
    slips = [(0, 1, 1, 1), (1, 1, 1, 1), (1, 1, 2, 1), (0, 1, 1, 1), (1, 1, 2, 1)] * 5
    assert route_prefix_from_trajectories(slips, target) == 1.0
    # wrong action in state 1: only state 0 counts
    leaves = [(0, 1, 1, 0), (1, 0, 2, 1)] * 5
    assert abs(route_prefix_from_trajectories(leaves, target) - 1 / 3) < 1e-9
    # wrong action in the starting state: nothing counts, however good the rest is
    wrong_start = [(0, 0, 1, 1), (1, 1, 2, 1)] * 5
    assert route_prefix_from_trajectories(wrong_start, target) == 0.0
    assert route_prefix_from_trajectories([], target) == 0.0


def test_hold_threshold_stops_perturbing_once_reward_is_reached():
    from builders import build_attack_env, build_encoders, build_population

    cfg = resolve_config({"env": "ring", "victim_n_episodes": 5, "hold_threshold": -1.0})
    population = build_population(cfg)
    obs_encoder, _ = build_encoders(population, cfg)
    env = build_attack_env(population, cfg, obs_encoder)
    env.reset()
    push = np.ones(env.action_space.shape)
    _, _, _, _, info = env.step(push)  # no previous reward yet: the push is applied
    assert info["held"] == 0.0
    sent = env._sent.copy()
    assert np.abs(sent).sum() > 0
    _, _, _, _, info = env.step(push)  # any reward is >= -1: held, nothing is sent
    assert info["held"] == 1.0
    assert np.array_equal(env._sent, sent)


def test_correction_scale_keeps_the_action_near_the_base():
    import torch

    from attack.ddpg import DDPG

    algo = DDPG(nb_states=4, nb_actions=3, max_action=1.0, correction_scale=0.2)
    with torch.no_grad():  # make the actor's output large
        for p in algo.actor.parameters():
            p.mul_(50.0)
    base = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    algo.set_base(base)
    for _ in range(5):
        action = algo.act_greedy(np.random.randn(4).astype(np.float32))
        assert np.abs(action - base).max() <= 0.2 + 1e-6
