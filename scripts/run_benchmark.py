#!/usr/bin/env python3
"""Run the SLAM binary over the benchmark sequences and report odometry metrics.
Usage:
    run_benchmark.py [--quick] [--sequences 03 limerock:fps10 ...]
                     [--baseline NAME] [--save-baseline NAME] [--label TEXT]
"""
import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO / "experiments" / "baselines"
DATASETS_FILE = REPO / "experiments" / "datasets.json"
DATASETS_TEMPLATE = """\
{
  "kitti": "~/data/kitti/dataset",
  "tracks": {"limerock": "~/data/limerock"},
  "sequences": ["03", "04", "06", "07", "limerock:fps10"],
  "quick": ["03", "limerock:fps10"]
}"""


def load_datasets():
    """Machine-local dataset locations and sequence sets; never committed."""
    if not DATASETS_FILE.is_file():
        sys.exit(f"no {DATASETS_FILE}\n"
                 f"Create it to point at your datasets, for example:\n{DATASETS_TEMPLATE}")
    config = json.loads(DATASETS_FILE.read_text())
    return {
        "kitti": Path(config["kitti"]).expanduser(),
        "tracks": {name: Path(path).expanduser()
                   for name, path in config.get("tracks", {}).items()},
        "sequences": config["sequences"],
        "quick": config.get("quick", config["sequences"]),
    }

# The evaluation scripts need numpy, which the system interpreter may not have
VENV_PYTHON = REPO / ".venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.is_file() else sys.executable


def git_revision():
    proc = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                          capture_output=True, text=True)
    return proc.stdout.strip() or "unknown"


def read_intrinsics(calib_path):
    """fx, fy, cx, cy from the P0 projection matrix of the left grayscale camera."""
    for line in open(calib_path):
        if line.startswith("P0:"):
            v = [float(x) for x in line.split()[1:]]
            return v[0], v[5], v[2], v[6]
    raise ValueError(f"no P0 row in {calib_path}")


def log_stats(log, runtime):
    text = log.read_text()
    matches = [int(m) for m in re.findall(r"Map matches with last frame: (\d+)", text)]
    return {
        "runtime": runtime,
        "frames": text.count("-" * 40),
        "key_frames": text.count("Adding key frame"),
        "matches": sum(matches) / len(matches) if matches else 0,
    }


def run_slam(binary, config, tag, workdir):
    """Runs one headless SLAM pass; returns (trajectory_path, log_stats) or an error string."""
    log = workdir / f"{tag}.log"
    start = time.time()
    with open(log, "w") as fh:
        proc = subprocess.run(
            [str(binary), str(config), "--headless",
             "--output-dir", str(workdir), "--run-id", tag, "--sequence", tag],
            stdout=fh, stderr=subprocess.STDOUT)
    runtime = time.time() - start
    out = workdir / tag / f"{tag}.txt"
    if proc.returncode != 0 or not out.is_file():
        return None, f"slam exited {proc.returncode}, see {log}"
    return (out, log_stats(log, runtime)), None


def render(ground_truth, estimate, tag):
    subprocess.run([PYTHON, str(REPO / "scripts" / "plot_trajectory.py"), str(ground_truth),
                    str(estimate), "-o", str(estimate.parent / f"{tag}.png"), "--title", tag],
                   capture_output=True)


def evaluate(script, ground_truth, estimate):
    proc = subprocess.run([PYTHON, str(REPO / "scripts" / script), str(ground_truth),
                           str(estimate)], capture_output=True, text=True)
    if proc.returncode != 0:
        return None, proc.stderr.strip().splitlines()[-1]
    return proc.stdout, None


def run_track(binary, tracks, track, subset, workdir):
    """Racing subsets ship their own config and ground truth next to the frames."""
    name = f"{track}:{subset}"
    source = tracks[track] / subset
    if not (source / "gt.txt").is_file():
        return {"seq": name, "error": f"no dataset at {source} (needs image_0/, gt.txt, "
                                      f"{track}.yaml); see docs on dataset preparation"}

    run, error = run_slam(binary, source / f"{track}.yaml", f"{track}_{subset or 'full'}",
                          workdir)
    if error:
        return {"seq": name, "error": error}
    out, stats = run

    output, error = evaluate("track_eval.py", source / "gt.txt", out)
    if error:
        return {"seq": name, "error": error}
    render(source / "gt.txt", out, f"{track}_{subset or 'full'}")

    result = {"seq": name, "r_err": float("nan"), **stats}
    for line in output.splitlines():
        # ATE as a percentage of path length is the comparable figure across subsets, and is
        # the closest thing to KITTI's translation error, so it goes in the same column
        if line.startswith("ATE as % of path:"):
            result["t_err"] = float(line.split()[-1])
        elif line.startswith("ATE (rmse):"):
            result["ate"] = float(line.split()[-2])
    return result


def run_kitti(binary, dataset, seq, workdir):
    images = dataset / "sequences" / seq / "image_0"
    ground_truth = dataset / "poses" / f"{seq}.txt"
    if not images.is_dir() or not ground_truth.is_file():
        return {"seq": seq, "error": f"no dataset at {dataset} (standard KITTI odometry "
                                     f"layout); download from the KITTI odometry benchmark"}

    fx, fy, cx, cy = read_intrinsics(dataset / "sequences" / seq / "calib.txt")
    config = workdir / f"kitti{seq}.yaml"
    config.write_text(f"video: {images}/%06d.png\nfx: {fx}\nfy: {fy}\ncx: {cx}\ncy: {cy}\n")

    run, error = run_slam(binary, config, f"kitti{seq}", workdir)
    if error:
        return {"seq": seq, "error": error}
    out, stats = run

    output, error = evaluate("kitti_eval.py", ground_truth, out)
    if error:
        return {"seq": seq, "error": error}
    render(ground_truth, out, f"kitti{seq}")

    t_err = r_err = float("nan")
    for line in output.splitlines():
        if line.startswith("Translation error:"):
            t_err = float(line.split()[2])
        elif line.startswith("Rotation error:"):
            r_err = float(line.split()[2])
    return {"seq": seq, "t_err": t_err, "r_err": r_err, **stats}


def run_sequence(binary, datasets, seq, workdir):
    if ":" in seq:
        track, subset = seq.split(":", 1)
        tracks = datasets["tracks"]
        if track not in tracks:
            return {"seq": seq, "error": f"unknown track {track}; known: {sorted(tracks)}"}
        return run_track(binary, tracks, track, subset, workdir)
    return run_kitti(binary, datasets["kitti"], seq, workdir)


def load_baseline(name):
    """Returns (name, sequences) for the named baseline, or the most recent one if unnamed."""
    if name:
        path = BASELINE_DIR / f"{name}.json"
        if not path.is_file():
            known = sorted(p.stem for p in BASELINE_DIR.glob("*.json"))
            sys.exit(f"no baseline named {name}; known: {known or 'none'}")
        return name, json.loads(path.read_text())["sequences"]
    candidates = sorted(BASELINE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None, {}
    return candidates[-1].stem, json.loads(candidates[-1].read_text())["sequences"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="the fast signal only: the quick set from datasets.json")
    parser.add_argument("--binary", type=Path, default=REPO / "build" / "slam")
    parser.add_argument("--label", default="",
                        help="tag for this configuration, printed in the header")
    parser.add_argument("--jobs", type=int, default=4, help="sequences to run concurrently")
    parser.add_argument("--baseline", default=None,
                        help="named baseline to diff against (default: most recently saved)")
    parser.add_argument("--save-baseline", default=None, metavar="NAME",
                        help="record these results as a named baseline")
    args = parser.parse_args()
    datasets = load_datasets()
    sequences = args.sequences or datasets["quick" if args.quick else "sequences"]

    baseline_name, baseline = load_baseline(args.baseline)
    run_id = args.save_baseline or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    workdir = REPO / "experiments" / "bench" / run_id
    workdir.mkdir(parents=True, exist_ok=True)

    if args.label:
        print(f"Configuration: {args.label}")
    print(f"Revision: {git_revision()}")
    print(f"Baseline: {baseline_name or 'none'}")
    print(f"Outputs:  {workdir}\n")
    header = (f"{'seq':<15} {'frames':>7} {'kf':>5} {'matches':>8} {'t_err %':>9} "
              f"{'vs base':>9} {'r_err deg/m':>12} {'runtime s':>10}")
    print(header)
    print("-" * len(header))

    # Sequences run concurrently: results do not depend on scheduling, so this only costs
    # oversubscription of the OpenCV thread pools and turns the total into the longest run
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_sequence, args.binary, datasets, seq, workdir): seq
                   for seq in sequences}
        by_seq = {futures[f]: f.result() for f in as_completed(futures)}

    for seq in sequences:
        result = by_seq.get(seq)
        if result is None or "error" in result:
            print(f"{seq:<15} {result['error'] if result else 'not run'}")
            continue
        results.append(result)
        was = baseline.get(seq, {}).get("t_err")
        delta = f"{result['t_err'] - was:+9.3f}" if was is not None else f"{'-':>9}"
        print(f"{result['seq']:<15} {result['frames']:>7} {result['key_frames']:>5} "
              f"{result['matches']:>8.0f} {result['t_err']:>9.3f} {delta} "
              f"{result['r_err']:>12.6f} {result['runtime']:>10.0f}")

    if results:
        print("-" * len(header))
        n = len(results)
        print(f"{'mean':<15} {'':>7} {'':>5} {sum(r['matches'] for r in results) / n:>8.0f} "
              f"{sum(r['t_err'] for r in results) / n:>9.3f} {'':>9} "
              f"{sum(r['r_err'] for r in results) / n:>12.6f} "
              f"{sum(r['runtime'] for r in results):>10.0f}")

    if args.save_baseline and results:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        path = BASELINE_DIR / f"{args.save_baseline}.json"
        path.write_text(json.dumps({
            "revision": git_revision(),
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "label": args.label,
            "sequences": {r["seq"]: {k: v for k, v in r.items() if k != "seq"}
                          for r in results},
        }, indent=2, sort_keys=True) + "\n")
        print(f"\nBaseline saved: {path}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
