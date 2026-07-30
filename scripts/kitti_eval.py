#!/usr/bin/env python3
"""Official KITTI odometry metrics: average translation error [%] and rotation error [deg/m].
Usage:
    kitti_eval.py <ground_truth.txt> <estimate.txt> [--no-align-scale]
"""
import argparse
import sys

import numpy as np

LENGTHS = [100, 200, 300, 400, 500, 600, 700, 800]
STEP_SIZE = 10


def load_poses(path):
    """Reads KITTI pose files: one 3x4 row-major camera-to-world matrix per line."""
    poses = []
    for line_no, line in enumerate(open(path), 1):
        if not line.strip():
            continue
        values = [float(v) for v in line.split()]
        if len(values) != 12:
            raise ValueError(f"{path}:{line_no}: expected 12 values, got {len(values)}")
        pose = np.eye(4)
        pose[:3, :4] = np.array(values).reshape(3, 4)
        poses.append(pose)
    return poses


def trajectory_distances(poses):
    """Cumulative path length at each pose."""
    dist = [0.0]
    for i in range(1, len(poses)):
        delta = poses[i][:3, 3] - poses[i - 1][:3, 3]
        dist.append(dist[i - 1] + float(np.linalg.norm(delta)))
    return dist


def last_frame_from_segment_length(dist, first_frame, length):
    """First frame beyond `length` metres from first_frame, or -1 if the trajectory ends first."""
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + length:
            return i
    return -1


def rotation_error(pose_error):
    """Rotation angle of the error matrix, in radians."""
    trace = pose_error[0, 0] + pose_error[1, 1] + pose_error[2, 2]
    d = 0.5 * (trace - 1.0)
    return float(np.arccos(max(min(d, 1.0), -1.0)))


def translation_error(pose_error):
    return float(np.linalg.norm(pose_error[:3, 3]))


def align_scale(poses_gt, poses_result):
    """Scales the estimate to best fit ground truth (least squares over positions)."""
    gt = np.array([p[:3, 3] for p in poses_gt])
    est = np.array([p[:3, 3] for p in poses_result])
    denom = float(np.sum(est * est))
    if denom == 0.0:
        return poses_result, 1.0
    scale = float(np.sum(gt * est)) / denom
    scaled = []
    for pose in poses_result:
        p = pose.copy()
        p[:3, 3] *= scale
        scaled.append(p)
    return scaled, scale


def calc_sequence_errors(poses_gt, poses_result):
    """Returns (t_err_per_m, r_err_per_m, segment_length) for every evaluated sub-sequence."""
    errors = []
    dist = trajectory_distances(poses_gt)
    for first_frame in range(0, len(poses_gt), STEP_SIZE):
        for length in LENGTHS:
            last_frame = last_frame_from_segment_length(dist, first_frame, length)
            if last_frame == -1:
                continue
            delta_gt = np.linalg.inv(poses_gt[first_frame]) @ poses_gt[last_frame]
            delta_result = np.linalg.inv(poses_result[first_frame]) @ poses_result[last_frame]
            pose_error = np.linalg.inv(delta_result) @ delta_gt
            errors.append((translation_error(pose_error) / length,
                           rotation_error(pose_error) / length,
                           length))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ground_truth")
    parser.add_argument("estimate")
    parser.add_argument("--no-align-scale", dest="align_scale", action="store_false",
                        help="skip scale alignment (for already-metric trajectories)")
    args = parser.parse_args()

    poses_gt = load_poses(args.ground_truth)
    poses_result = load_poses(args.estimate)
    if len(poses_gt) != len(poses_result):
        print(f"Pose count mismatch: ground truth has {len(poses_gt)}, "
              f"estimate has {len(poses_result)}", file=sys.stderr)
        return 1

    scale = 1.0
    if args.align_scale:
        poses_result, scale = align_scale(poses_gt, poses_result)

    errors = calc_sequence_errors(poses_gt, poses_result)
    if not errors:
        total = trajectory_distances(poses_gt)[-1]
        print(f"No sub-sequences of {LENGTHS[0]} m or more: trajectory is only {total:.1f} m",
              file=sys.stderr)
        return 1

    t_err = float(np.mean([e[0] for e in errors]))
    r_err = float(np.mean([e[1] for e in errors]))

    print(f"Sub-sequences evaluated: {len(errors)}")
    print(f"Path length:             {trajectory_distances(poses_gt)[-1]:.1f} m")
    if args.align_scale:
        print(f"Scale correction:        {scale:.4f}")
    print(f"Translation error:       {t_err * 100:.4f} %")
    print(f"Rotation error:          {r_err * 180 / np.pi:.6f} deg/m")

    print("\nPer segment length:")
    for length in LENGTHS:
        subset = [e for e in errors if e[2] == length]
        if not subset:
            continue
        t = float(np.mean([e[0] for e in subset])) * 100
        r = float(np.mean([e[1] for e in subset])) * 180 / np.pi
        print(f"  {length:3d} m  ({len(subset):3d} segments)  "
              f"t_err = {t:6.3f} %   r_err = {r:.6f} deg/m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
