#!/usr/bin/env python3
"""Benchmarks of n2p2 builds: LAMMPS MD and nnp-train.

Runs an example once per configuration and reports the timing, so that CPU
and GPU builds (or different numbers of ranks, GPUs, ...) can be compared on
the same machine. Nothing here depends on a particular cluster: binaries,
the MPI launcher and the work directory are arguments; call the script from
your own job script.

  md     LAMMPS example (md.lmp with "variable numSteps" and
         "run ${numSteps}"), replicated in all three directions: a warm-up
         run, then the timed run; reports ns/day and the loop time.
  train  nnp-train example (input.nn, input.data, scaling.data): the given
         number of epochs; reports the time of each epoch.

Results are printed as a table and can be appended to a CSV file.
See README.md in this directory.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import tempfile
import sys
import time


def start_mps(workdir):
    """Start an MPS daemon for this run, return its environment.

    The pipe directory holds the daemon's Unix sockets, whose paths are
    limited to about 100 characters: it goes to the temporary directory
    (TMPDIR or /tmp), not to the (possibly long) work directory, which keeps
    the log directory.
    """
    env = dict(os.environ)
    tag = "%d_%d" % (os.getpid(), int(time.time()))
    env["CUDA_MPS_PIPE_DIRECTORY"] = tempfile.mkdtemp(prefix="n2p2_mps_")
    env["CUDA_MPS_LOG_DIRECTORY"] = os.path.join(workdir, "mps_log_" + tag)
    os.makedirs(env["CUDA_MPS_LOG_DIRECTORY"], exist_ok=True)
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)
    return env


def stop_mps(env):
    subprocess.run(["nvidia-cuda-mps-control"], input=b"quit\n", env=env)
    shutil.rmtree(env["CUDA_MPS_PIPE_DIRECTORY"], ignore_errors=True)


def launch(args, ranks, command, rundir, env):
    """Run command on ranks MPI processes in rundir, return (rc, seconds)."""
    launcher = args.mpirun.format(np=ranks).split()
    if args.mps:
        # OpenMPI: forward the MPS variables to all ranks.
        if os.path.basename(launcher[0]) == "mpirun":
            launcher += ["-x", "CUDA_MPS_PIPE_DIRECTORY",
                         "-x", "CUDA_MPS_LOG_DIRECTORY"]
    start = time.time()
    with open(os.path.join(rundir, "out.log"), "w") as out:
        rc = subprocess.run(launcher + command, cwd=rundir, env=env,
                            stdout=out, stderr=subprocess.STDOUT).returncode
    return rc, time.time() - start


def copy_example(example, rundir):
    if os.path.exists(rundir):
        shutil.rmtree(rundir)
    shutil.copytree(example, rundir)


def md_input(text, replicate, steps, warmup):
    """md.lmp for a benchmark: replicated, no dumps, warm-up + timed run."""
    out = []
    for line in text.split("\n"):
        stripped = line.strip()
        if re.match(r"variable\s+numSteps\s", stripped):
            line = "variable numSteps equal %d" % steps
        elif re.match(r"(dump|dump_modify|restart)\s", stripped):
            continue
        elif re.match(r"thermo\s+\d+\s*$", stripped):
            line = "thermo %d" % max(steps // 10, 1)
        elif re.match(r"run\s+\$\{numSteps\}", stripped):
            if warmup > 0:
                out.append("run %d" % warmup)
        out.append(line)
        if re.match(r"read_data\s", stripped) and replicate > 1:
            out.append("replicate %d %d %d" % (replicate, replicate,
                                               replicate))
    return "\n".join(out)


def lammps_results(logfile):
    """Atoms, loop time and ns/day of the last run in a LAMMPS log."""
    atoms = loop = nsday = None
    with open(logfile) as f:
        for line in f:
            m = re.match(r"Loop time of (\S+) on \d+ procs for (\d+) steps "
                         r"with (\d+) atoms", line)
            if m:
                loop, atoms = float(m.group(1)), int(m.group(3))
            m = re.match(r"Performance: (\S+) ns/day", line)
            if m:
                nsday = float(m.group(1))
    return atoms, loop, nsday


def run_md(args):
    rows = []
    env = start_mps(args.workdir) if args.mps else dict(os.environ)
    try:
        for rep in args.replicate:
            steps = args.steps if args.steps else max(200 // rep ** 3, 10)
            rundir = os.path.join(args.workdir, "%s_md_r%d" % (args.label,
                                                                rep))
            copy_example(args.example, rundir)
            path = os.path.join(rundir, args.input)
            with open(path) as f:
                text = f.read()
            with open(path, "w") as f:
                f.write(md_input(text, rep, steps, args.warmup))
            rc, wall = launch(args, args.ranks,
                              [args.lmp, "-in", args.input], rundir, env)
            log = os.path.join(rundir, "log.lammps")
            atoms, loop, nsday = (lammps_results(log) if os.path.exists(log)
                                  else (None, None, None))
            rows.append({"label": args.label, "benchmark": "md",
                         "replicate": rep, "atoms": atoms, "ranks": args.ranks,
                         "steps": steps, "loop_s": loop, "ns_per_day": nsday,
                         "wall_s": round(wall, 1), "rc": rc})
    finally:
        if args.mps:
            stop_mps(env)
    return rows


def run_train(args):
    rundir = os.path.join(args.workdir, "%s_train" % args.label)
    copy_example(args.example, rundir)
    nn = os.path.join(rundir, "input.nn")
    with open(nn) as f:
        text = f.read()
    text = re.sub(r"(?m)^\s*epochs\s+\d+", "epochs %d" % args.epochs, text)
    with open(nn, "w") as f:
        f.write(text)
    env = start_mps(args.workdir) if args.mps else dict(os.environ)
    try:
        command = [args.nnp_train] + ([str(args.stage)] if args.stage else [])
        rc, wall = launch(args, args.ranks, command, rundir, env)
    finally:
        if args.mps:
            stop_mps(env)
    epochs = []
    logs = sorted(name for name in os.listdir(rundir)
                  if name.startswith("nnp-train.log.0000"))
    if logs:
        with open(os.path.join(rundir, logs[0])) as f:
            for line in f:
                t = line.split()
                # TIMING <epoch> <count> <train> <error> <other> <epoch time>
                if len(t) == 8 and t[0] == "TIMING" and t[1].isdigit() \
                   and int(t[1]) > 0:
                    epochs.append(float(t[6]))
    return [{"label": args.label, "benchmark": "train", "ranks": args.ranks,
             "epochs": len(epochs),
             "epoch_s_mean": (round(sum(epochs) / len(epochs), 2)
                              if epochs else None),
             "epoch_s": " ".join("%.2f" % e for e in epochs),
             "wall_s": round(wall, 1), "rc": rc}]


def report(rows, csvfile):
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows))
              for k in keys}
    print("  ".join(k.ljust(widths[k]) for k in keys))
    for row in rows:
        print("  ".join(str(row.get(k, "")).ljust(widths[k]) for k in keys))
    if csvfile:
        new = not os.path.exists(csvfile)
        with open(csvfile, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if new:
                writer.writeheader()
            writer.writerows(rows)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="benchmark")
    sub.required = True
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--example", required=True,
                        help="example directory (copied for each run)")
    common.add_argument("--ranks", type=int, default=1,
                        help="number of MPI processes")
    common.add_argument("--mpirun", default="mpirun -np {np}",
                        help="launcher, {np} is replaced by the number of "
                             "processes (default: %(default)s)")
    common.add_argument("--mps", action="store_true",
                        help="start an NVIDIA MPS daemon for the run (several "
                             "ranks per GPU)")
    common.add_argument("--workdir", default=".",
                        help="directory for the run directories")
    common.add_argument("--label", default="run",
                        help="name of this configuration in the results")
    common.add_argument("--csv", help="append the results to this CSV file")
    md = sub.add_parser("md", parents=[common], help="LAMMPS MD")
    md.add_argument("--lmp", required=True, help="LAMMPS executable")
    md.add_argument("--input", default="md.lmp", help="LAMMPS input file")
    md.add_argument("--replicate", type=int, nargs="+", default=[1],
                    help="replication factors (in each direction)")
    md.add_argument("--steps", type=int, default=0,
                    help="timed steps (default: 200 / replicate^3, at "
                         "least 10)")
    md.add_argument("--warmup", type=int, default=10,
                    help="steps run before the timed run")
    train = sub.add_parser("train", parents=[common], help="nnp-train")
    train.add_argument("--nnp-train", required=True,
                       help="nnp-train executable")
    train.add_argument("--epochs", type=int, default=2,
                       help="number of epochs")
    train.add_argument("--stage", type=int, default=0,
                       help="training stage (4G: 1 or 2)")
    args = parser.parse_args(argv)
    args.example = os.path.abspath(args.example)
    args.workdir = os.path.abspath(args.workdir)
    os.makedirs(args.workdir, exist_ok=True)
    rows = run_md(args) if args.benchmark == "md" else run_train(args)
    report(rows, args.csv)
    return 0 if all(r["rc"] == 0 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
