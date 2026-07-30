#!/usr/bin/env python3
"""Absolute trajectory error against the Lime Rock ground truth.
Usage:
    track_eval.py <ground_truth.txt> <estimate.txt>
"""
import sys

import numpy as np


def read_positions(path):
    poses = np.loadtxt(path).reshape(-1, 3, 4)
    return poses[:, :, 3], poses[:, :, :3]


def align_start(source, target):
    """Rotation and scale about the start point taking source onto target. Both trajectories
    are anchored at their start (no free translation, since drift at the start is zero by
    construction); rotation is free because there is no defined initial heading."""
    u, s, vt = np.linalg.svd(target.T @ source / len(source))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ d @ vt
    scale = float(np.trace(np.diag(s) @ d) / (source**2).sum() * len(source))
    return scale, rotation


def heading(rotations):
    """Total yaw swept, in degrees, accumulated so wrapping past 180 does not fold back."""
    yaw = np.unwrap([np.arctan2(r[0, 2], r[2, 2]) for r in rotations])
    return float(np.degrees(yaw[-1] - yaw[0]))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    gt_positions, gt_rotations = read_positions(sys.argv[1])
    positions, rotations = read_positions(sys.argv[2])

    n = min(len(gt_positions), len(positions))
    if n < len(gt_positions):
        print(f"Estimate is short: {len(positions)} of {len(gt_positions)} poses")
    gt_positions, gt_rotations = gt_positions[:n], gt_rotations[:n]
    positions, rotations = positions[:n], rotations[:n]

    gt_positions = gt_positions - gt_positions[0]
    positions = positions - positions[0]
    scale, rotation = align_start(positions, gt_positions)
    aligned = (scale * rotation @ positions.T).T

    error = np.linalg.norm(aligned - gt_positions, axis=1)
    path = float(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1).sum())
    print(f"Poses:               {n}")
    print(f"Path length:         {path:.1f} m")
    print(f"Scale correction:    {scale:.4f}")
    print(f"ATE (rmse):          {np.sqrt((error**2).mean()):.2f} m")
    print(f"ATE (max):           {error.max():.2f} m")
    print(f"ATE as % of path:    {100 * np.sqrt((error**2).mean()) / path:.2f}")
    print(f"Endpoint gap:        {error[-1]:.2f} m")
    print(f"Heading gt:          {heading(gt_rotations):+.1f} deg")
    print(f"Heading estimate:    {heading(rotations):+.1f} deg")
    print(f"Heading error:       {heading(rotations) - heading(gt_rotations):+.1f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
