import os
import time
import xml.etree.ElementTree as ET


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


def wait_until_idle(num_req, thresh=0.01, verbose=False, candidate=None):
    while True:
        available = []
        stat = get_status(thresh=thresh)
        for each in stat:
            if each["ready"]:
                available.append(each["gpu_id"])
        if len(available) >= num_req:
            if candidate:
                in_candidate = [e for e in available if int(e) in candidate]
                if len(in_candidate) >= num_req:
                    return in_candidate[:num_req]
            else:
                return available[:num_req]
        time.sleep(20)
        if verbose:
            print("Waiting ")
