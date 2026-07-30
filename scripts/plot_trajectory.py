#!/usr/bin/env python3
"""Top down plot of an estimated trajectory against ground truth.
Usage:
    plot_trajectory.py <ground_truth.txt> <estimate.txt> [-o out.png] [--title TEXT]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_positions(path):
    return np.loadtxt(path).reshape(-1, 3, 4)[:, :, 3]


def umeyama(source, target):
    source_mean, target_mean = source.mean(0), target.mean(0)
    a, b = source - source_mean, target - target_mean
    u, s, vt = np.linalg.svd(b.T @ a / len(a))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ d @ vt
    scale = float(np.trace(np.diag(s) @ d) / (a**2).sum() * len(a))
    return scale, rotation, target_mean - scale * rotation @ source_mean


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("estimate", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=Path("trajectory.png"))
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    gt = read_positions(args.ground_truth)
    est = read_positions(args.estimate)
    n = min(len(gt), len(est))
    gt, est = gt[:n], est[:n]

    scale, rotation, translation = umeyama(est, gt)
    aligned = (scale * rotation @ est.T).T + translation
    error = np.linalg.norm(aligned - gt, axis=1)
    distance = np.concatenate(
        [[0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))]
    )

    span = np.ptp(np.vstack([gt, aligned]), axis=0)
    height = float(np.clip(6.5 * span[2] / max(span[0], 1e-6), 5.0, 11.0))
    map_width = float(np.clip(height * span[0] / max(span[2], 1e-6), 3.0, 9.0))
    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(map_width + 7.0, height),
        gridspec_kw={"width_ratios": [map_width, 7.0]},
    )

    # KITTI's x is lateral and z is forward, so a top down view is x against z
    left.plot(gt[:, 0], gt[:, 2], color="0.35", linewidth=2.4, label="ground truth")
    left.plot(
        aligned[:, 0], aligned[:, 2], color="#c0392b", linewidth=1.6, label="SLAM"
    )
    left.scatter(*gt[0, [0, 2]], s=90, color="#27ae60", zorder=5, label="start")
    left.scatter(
        *gt[-1, [0, 2]], s=90, marker="s", color="0.35", zorder=5, label="end (truth)"
    )
    left.scatter(
        *aligned[-1, [0, 2]],
        s=90,
        marker="s",
        color="#c0392b",
        zorder=5,
        label="end (SLAM)",
    )
    left.set_aspect("equal")
    left.set_xlabel("x [m]")
    left.set_ylabel("z [m]")
    left.legend(loc="best", frameon=False)
    left.grid(alpha=0.25)
    left.set_title(args.title or args.estimate.stem)

    right.plot(distance, error, color="#c0392b", linewidth=1.6)
    right.set_xlabel("distance travelled [m]")
    right.set_ylabel("position error [m]")
    right.grid(alpha=0.25)
    right.set_title(
        f"rmse {np.sqrt((error**2).mean()):.1f} m, "
        f"final {error[-1]:.1f} m, scale x{scale:.3f}"
    )

    figure.tight_layout()
    figure.savefig(args.out, dpi=130)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
