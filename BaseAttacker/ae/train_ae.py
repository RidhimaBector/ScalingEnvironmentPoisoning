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
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ae.ae import AutoEncoder, default_ae_name, device
from envs.env3D_4x4 import Grid3D
from victim.victim_Q import VictimQLearning

UNVISITED = 4  # class index used for "state not visited yet" (trace value -1)


def trace_to_actions(trace: np.ndarray) -> np.ndarray:
    a = trace[:, 1].copy()
    a[a < 0] = UNVISITED
    return a


def collect(n_attack_episodes: int, steps: int, victim_eps: int, seed: int, shape=(4, 4)) -> np.ndarray:
    """Return an (N, nS) int array of per-state last actions (4 = unvisited)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    env = Grid3D(grid_dimensions=tuple(shape))
    out = []
    for ep in range(n_attack_episodes):
        env.reset_dynamics()
        victim = VictimQLearning(env.nS, env.nA, memory_size=50)
        goal_alt = rng.uniform(0, 10, size=env.nS) if ep % 2 == 0 else None
        for _ in range(steps):
            if goal_alt is not None:
                act = np.clip(goal_alt - env.altitude.flatten(), -1, 1)
            else:
                act = rng.uniform(-1, 1, size=env.nS)
            env.apply_perturbation(act)
            victim.train(env, victim_eps)
            out.append(trace_to_actions(victim.get_behavior_trace()))
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
    ap.add_argument('--grid_size', type=int, default=4, help='square grid N x N')
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
    rows, cols = args.shape or (args.grid_size, args.grid_size)
    is_4x4 = (rows, cols) == (4, 4)
    if args.out is None:
        args.out = os.path.join('ae', 'models', default_ae_name(rows, cols))
    emb = args.embedding or (5 if is_4x4 else rows * cols // 3)
    hidden = args.hidden or (36 if is_4x4 else 2 * rows * cols)
    data = collect(args.attack_episodes, args.steps, args.victim_eps, args.seed, (rows, cols))
    perm = np.random.default_rng(args.seed).permutation(len(data))
    n_val = len(data) // 10
    val, train = data[perm[:n_val]], data[perm[n_val:]]
    print(f"collected {len(data)} traces  train={len(train)} val={len(val)}  "
          f"unvisited frac={np.mean(data == UNVISITED):.3f}")

    nS = data.shape[1]
    ae = AutoEncoder(enc_in_size=nS * 2, enc_out_size=emb, dec_in_size=emb + 1, dec_out_size=5,
                     lr=args.lr, hidden=hidden)
    majority = max(np.mean(val == k) for k in range(5))
    print(f"grid {rows}x{cols}  embedding={emb}  hidden={hidden}  majority-class baseline={majority:.3f}")
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
