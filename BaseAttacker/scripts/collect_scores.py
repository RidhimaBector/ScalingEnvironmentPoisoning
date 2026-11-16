"""Collect score.json files from finished runs into one table.

    python scripts/collect_scores.py results/sweep-3x3x3/*            # one row per run
    python scripts/collect_scores.py results/sweep-3x3x3/* --by cell  # mean over seeds

Columns: target_match and route are ground truth on fresh victims, shown as
"no attack -> attacked". Runs are named by naming.py and end in __seed<N>.
"""
import argparse
import json
import os
import re
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--by", choices=["run", "cell"], default="run")
    a = ap.parse_args()

    rows = []
    for d in sorted(a.run_dirs):
        path = os.path.join(d, "score.json")
        if not os.path.exists(path):
            continue
        s = json.load(open(path))
        name = os.path.basename(d.rstrip("/"))
        rows.append((re.sub(r"__seed\d+$", "", name), name, s))

    print(f"{'run':40s} {'victims':>7s} {'target_match':>15s} {'route':>13s} {'victim_return':>17s}")
    if a.by == "run":
        for _cell, name, s in rows:
            n, b, t = s["episodes"], s["baseline"], s["attacked"]
            print(f"{name:40s} {n:7d} {b['greedy_accuracy']:6.2f} -> {t['greedy_accuracy']:4.2f}"
                  f" {s['baseline_route_count']:4d} -> {s['route_count']:3d}/{n:<2d}"
                  f" {b['victim_return']:7.1f} -> {t['victim_return']:6.1f}")
        return
    cells = defaultdict(list)
    for cell, _name, s in rows:
        cells[cell].append(s)
    for cell, group in cells.items():
        n = sum(s["episodes"] for s in group)
        mean = lambda cond, key: sum(s[cond][key] for s in group) / len(group)  # noqa: E731
        print(f"{cell + f' ({len(group)} seeds)':40s} {n:7d}"
              f" {mean('baseline', 'greedy_accuracy'):6.2f} -> {mean('attacked', 'greedy_accuracy'):4.2f}"
              f" {sum(s['baseline_route_count'] for s in group):4d} -> {sum(s['route_count'] for s in group):3d}/{n:<3d}"
              f" {mean('baseline', 'victim_return'):7.1f} -> {mean('attacked', 'victim_return'):6.1f}")


if __name__ == "__main__":
    main()
