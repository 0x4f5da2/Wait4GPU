import xml.etree.ElementTree as ET
import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER
import time


def get_status(thresh=0.01):
    s = os.popen("/usr/bin/nvidia-smi -q -x")
    xml = s.read()
    xml_root = ET.fromstring(xml)
    all_gpu = xml_root.findall("./gpu")
    gpu_stat = []
    for each in all_gpu:
        d = dict()
        d["gpu_id"] = each.find("./minor_number").text
        d["mem_all"] = int(each.find("./fb_memory_usage/total").text[:-3])
        d["mem_used"] = int(each.find("./fb_memory_usage/used").text[:-3])
        d["ready"] = d["mem_used"] / d["mem_all"] < thresh
        gpu_stat.append(d)
    return gpu_stat


def wait_until_idle(num_req, thresh=0.01, verbose=False):
    while True:
        available = []
        stat = get_status(thresh=thresh)
        for each in stat:
            if each["ready"]:
                available.append(each["gpu_id"])
        if len(available) >= num_req:
            return available[:num_req]
        time.sleep(20)
        if verbose:
            print("Waiting ")


def parse_args():

    parser = ArgumentParser(description="Launch script when gpu is idle.")

    parser.add_argument("--num-required", type=int, default=1, help="Num GPUs required for your script")

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

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    gpus = wait_until_idle(args.num_required)
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
