import os
import subprocess
import sys
from argparse import ArgumentParser, REMAINDER

from .wait4gpu import wait_until_idle


def parse_args():
    parser = ArgumentParser(description="Launch script when gpu is idle.")

    parser.add_argument("--num-required", type=int, default=1, help="Num GPUs required for your script")

    parser.add_argument("--candidate", type=str, default="", help="Candidate GPUs")

    parser.add_argument("--threshold", type=float, default=0.02, help="Threshold for available GPUs")

    parser.add_argument("--no-python", default=False, action="store_true",
                        help="Do not prepend the training script with \"python\" - just exec "
                             "it directly. Useful when the script is not a Python script.")

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the GPU training program/script to be launched")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # set environmental variables
    current_env = os.environ.copy()
    candidate = None if not args.candidate else list(map(int, args.candidate.split(",")))
    gpus = wait_until_idle(args.num_required, candidate=candidate, thresh=args.threshold)
    current_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

    # spawn the processes
    with_python = not args.no_python
    cmd = []
    if with_python:
        cmd = [sys.executable, "-u"]

    cmd.append(args.training_script)

    cmd.extend(args.training_script_args)

    process = subprocess.Popen(cmd, env=current_env)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode,
                                            cmd=cmd)


if __name__ == "__main__":
    main()
