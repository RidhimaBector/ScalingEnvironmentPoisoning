"""Train the behaviour-trace AutoEncoder on REAL victim traces.

The shipped weights (models/340240_f-o_AutoEncoder_SftMx) were trained on
uniformly random action tables. This script collects behaviour traces from
Q-learning victims while the altitude is perturbed (random walks plus pushes
toward random altitude maps), then trains the same AutoEncoder architecture to
reconstruct the per-state last action from the 5-D embedding.

Usage (from BaseAttacker/):
    python -m ae.train_ae --out ae/models/real_traces --epochs 60
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ae.ae import AutoEncoder, default_ae_name, device
from registry import ENV_MAP, VICTIM_ALGO_MAP


def trace_to_actions(trace: np.ndarray, unvisited: int) -> np.ndarray:
    """Per-state last action; states not visited yet (-1) get the extra class `unvisited`."""
    a = trace[:, 1].copy()
    a[a < 0] = unvisited
    return a


def collect(env, victim_class, n_attack_episodes: int, steps: int, victim_eps: int, seed: int) -> np.ndarray:
    """Return an (N, nS) int array of per-state last actions (nA = unvisited).

    Works for any VictimEnvironment: perturbations are drawn from env.perturbation_space.
    Half the attack episodes random-walk; the other half steer toward a random point
    inside env.dynamics_bounds() (or hold one random direction when there are no bounds),
    so the traces cover victims adapting to very different dynamics.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    space = env.perturbation_space
    low, high = np.asarray(space.low, dtype=float), np.asarray(space.high, dtype=float)
    bounds = env.dynamics_bounds() if hasattr(env, 'dynamics_bounds') else None
    same_dim = env.dynamics_dim == int(np.prod(space.shape))
    out = []
    for ep in range(n_attack_episodes):
        env.reset_dynamics()
        victim = victim_class(env.nS, env.nA)
        steer = ep % 2 == 0
        goal = rng.uniform(bounds[0], bounds[1], size=space.shape) if (steer and bounds and same_dim) else None
        direction = rng.choice([-1.0, 0.0, 1.0], size=space.shape)
        for _ in range(steps):
            if goal is not None:
                act = np.clip(goal - np.asarray(env.get_dynamics()).reshape(space.shape), low, high)
            elif steer:
                act = np.clip(direction * high, low, high)
            else:
                act = rng.uniform(low, high, size=space.shape)
            env.apply_perturbation(act)
            victim.train(env, victim_eps)
            out.append(trace_to_actions(victim.get_behavior_trace(), env.nA))
    return np.asarray(out, dtype=np.int64)


def batch_tensors(actions: np.ndarray):
    n, nS = actions.shape
    states = np.tile(np.arange(nS, dtype=np.float32)[None, :, None], (n, 1, 1))
    acts = actions.astype(np.float32)[:, :, None]
    x = torch.as_tensor(np.concatenate([states, acts], axis=2), device=device)
    return x, torch.as_tensor(states, device=device), torch.as_tensor(actions, device=device).reshape(-1)


def recon_accuracy(ae: AutoEncoder, actions: np.ndarray) -> float:
    ae.Model.eval()
    with torch.no_grad():
        x, states, tgt = batch_tensors(actions)
        pred = ae.Model(x, actions.shape[1], states).argmax(dim=1)
    return float((pred == tgt).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None, help='default: ae/models/real_traces[_NxN]')
    ap.add_argument('--env', default='grid3d', choices=sorted(ENV_MAP), help='victim environment')
    ap.add_argument('--victim_algo', default='qlearning', choices=sorted(VICTIM_ALGO_MAP))
    ap.add_argument('--env_kwargs', default='{}', help='JSON dict of environment constructor arguments')
    ap.add_argument('--grid_size', type=int, default=4, help='grid3d only: square grid N x N')
    ap.add_argument('--shape', type=int, nargs=2, default=None, metavar=('ROWS', 'COLS'),
                    help='rectangular grid; overrides --grid_size')
    ap.add_argument('--embedding', type=int, default=None, help='default: 5 for 4x4, else nS // 3')
    ap.add_argument('--hidden', type=int, default=None, help='default: 36 for 4x4, else 2 * nS')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--attack_episodes', type=int, default=600)
    ap.add_argument('--steps', type=int, default=15)
    ap.add_argument('--victim_eps', type=int, default=80)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init', default=None, help='optional weights to fine-tune from')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.env == 'grid3d':
        env = ENV_MAP['grid3d'](grid_dimensions=tuple(args.shape or (args.grid_size, args.grid_size)))
    else:
        env = ENV_MAP[args.env](**json.loads(args.env_kwargs))
    nS, nA = int(env.nS), int(env.nA)
    small = nS <= 16
    if args.out is None:
        args.out = os.path.join('ae', 'models', default_ae_name(env))
    emb = args.embedding or (5 if small else nS // 3)
    hidden = args.hidden or (36 if small else 2 * nS)
    data = collect(env, VICTIM_ALGO_MAP[args.victim_algo], args.attack_episodes, args.steps,
                   args.victim_eps, args.seed)
    perm = np.random.default_rng(args.seed).permutation(len(data))
    n_val = len(data) // 10
    val, train = data[perm[:n_val]], data[perm[n_val:]]
    print(f"collected {len(data)} traces  train={len(train)} val={len(val)}  "
          f"unvisited frac={np.mean(data == nA):.3f}")

    ae = AutoEncoder(enc_in_size=nS * 2, enc_out_size=emb, dec_in_size=emb + 1, dec_out_size=nA + 1,
                     lr=args.lr, hidden=hidden)
    majority = max(np.mean(val == k) for k in range(nA + 1))
    print(f"{type(env).__name__} nS={nS} nA={nA}  embedding={emb}  hidden={hidden}  "
          f"majority-class baseline={majority:.3f}")
    if args.init:
        ae.load(args.init)
        print(f"init from {args.init}: val recon acc = {recon_accuracy(ae, val):.3f}")
    opt = torch.optim.Adam(ae.Model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        ae.Model.train()
        order = np.random.permutation(len(train))
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            b = train[order[i:i + args.batch_size]]
            x, states, tgt = batch_tensors(b)
            probs = ae.Model(x, nS, states)  # decoder already applies softmax
            loss = torch.nn.functional.nll_loss(torch.log(probs + 1e-8), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train nll={total / len(train):.4f}  "
                  f"train acc={recon_accuracy(ae, train[:2000]):.3f}  val acc={recon_accuracy(ae, val):.3f}")

    ae.save(args.out)
    print(f"saved {args.out}_AutoEncoder   final val recon acc={recon_accuracy(ae, val):.3f}")


if __name__ == '__main__':
    main()
