"""Efficiency benchmark: parameters, model size, latency and peak memory per variant.

Each measurement runs in a fresh subprocess so peak memory is not polluted by
earlier models.

  python -m scripts.benchmark --teacher runs/teacher/best.pt --student runs/student/best.pt \
      --onnx weights/student_fp32.onnx weights/student_int8.onnx --size 1024
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np


def peak_rss_mb() -> float:
    import psutil

    mi = psutil.Process().memory_info()
    if hasattr(mi, "peak_wset"):  # Windows
        return mi.peak_wset / 1e6
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3  # Linux: KB


def rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1e6


def worker(job: dict) -> dict:
    size, runs = job["size"], job["runs"]
    kind = job["kind"]
    base = rss_mb()
    res = {"name": job["name"], "kind": kind, "size": size}
    if kind == "torch":
        import torch

        from models.nafnet import count_params, run_padded
        from models.zoo import load_ckpt

        dev = job["device"]
        m = load_ckpt(job["path"], dev)
        if job.get("half"):
            m = m.half()
        res["params_M"] = count_params(m) / 1e6
        res["size_MB"] = res["params_M"] * (2 if job.get("half") else 4)
        s = size // m.out_shuffle
        x = torch.rand(1, m.in_ch, s, s, device=dev, dtype=torch.float16 if job.get("half") else torch.float32)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(3):
                run_padded(m, x)
            ts = []
            for _ in range(runs):
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                run_padded(m, x)
                if dev == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        if dev == "cuda":
            res["gpu_peak_MB"] = torch.cuda.max_memory_allocated() / 1e6
            res["device"] = torch.cuda.get_device_name(0)
        else:
            res["device"] = platform.processor() or "cpu"
    elif kind == "onnx":
        import onnx
        import onnxruntime as ort

        so = ort.SessionOptions()
        if job.get("threads"):
            so.intra_op_num_threads = job["threads"]
        sess = ort.InferenceSession(job["path"], so, providers=["CPUExecutionProvider"])
        g = onnx.load(job["path"])
        res["params_M"] = sum(int(np.prod(t.dims)) for t in g.graph.initializer) / 1e6
        res["size_MB"] = os.path.getsize(job["path"]) / 1e6
        cin = sess.get_inputs()[0].shape[1]
        x = np.random.rand(1, cin, size, size).astype(np.float32)
        for _ in range(2):
            sess.run(None, {"input": x})
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            sess.run(None, {"input": x})
            ts.append(time.perf_counter() - t0)
        res["device"] = f"CPU ({os.cpu_count()} threads avail, ORT intra={job.get('threads') or 'default'})"
    elif kind == "pipeline":  # full app path: classical render + router, ONNX student
        from isp.hybrid import HybridISP
        from isp.raw_io import from_srgb
        from isp.router import OnnxRestorer

        rest = {"student": OnnxRestorer(job["path"], "student", threads=job.get("threads"))} if job.get("path") else {}
        h = HybridISP(rest)
        img = np.random.default_rng(0).random((size, size, 3)).astype(np.float32)
        raw = from_srgb(img, 0.5)
        h.process(raw, job.get("budget", "student_only"))
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            h.process(raw, job.get("budget", "student_only"))
            ts.append(time.perf_counter() - t0)
        res["device"] = "CPU"
        res["budget"] = job.get("budget")
    res["latency_ms_median"] = float(np.median(ts) * 1e3)
    res["latency_ms_p90"] = float(np.percentile(ts, 90) * 1e3)
    res["megapixels"] = size * size / 1e6
    res["peak_rss_MB"] = peak_rss_mb()
    res["rss_delta_MB"] = rss_mb() - base
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher"), p.add_argument("--student"), p.add_argument("--e2e")
    p.add_argument("--onnx", nargs="*", default=[])
    p.add_argument("--app-onnx", default=None, help="ONNX used by the app for the full-pipeline timing")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--threads", type=int, default=2, help="CPU threads (HF Spaces free tier = 2 vCPU)")
    p.add_argument("--out", default="results/benchmark.json")
    p.add_argument("--job", default=None, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.job:
        print("RESULT " + json.dumps(worker(json.loads(a.job))))
        return

    import torch

    gpu = torch.cuda.is_available()
    jobs = []
    for name, path in [("teacher", a.teacher), ("student", a.student), ("e2e", a.e2e)]:
        if not path:
            continue
        if gpu:
            jobs.append({"name": f"{name}_gpu_fp32", "kind": "torch", "path": path, "device": "cuda"})
            jobs.append({"name": f"{name}_gpu_fp16", "kind": "torch", "path": path, "device": "cuda", "half": True})
        if name == "student" or not gpu:
            jobs.append({"name": f"{name}_cpu_torch_fp32", "kind": "torch", "path": path, "device": "cpu"})
    for path in a.onnx:
        jobs.append({"name": f"onnx_{os.path.basename(path)}_cpu{a.threads}t", "kind": "onnx", "path": path,
                     "threads": a.threads})
    jobs.append({"name": "pipeline_classical_only_cpu", "kind": "pipeline", "path": None, "budget": "classical_only"})
    if a.app_onnx:
        for b in ("student_only", "low"):
            jobs.append({"name": f"pipeline_{b}_cpu{a.threads}t", "kind": "pipeline", "path": a.app_onnx, "budget": b,
                         "threads": a.threads})
    results = []
    for j in jobs:
        j.setdefault("size", a.size)
        j.setdefault("runs", a.runs if j["kind"] != "pipeline" else max(3, a.runs // 4))
        cp = subprocess.run([sys.executable, "-m", "scripts.benchmark", "--job", json.dumps(j)], capture_output=True,
                            text=True)
        line = [ln for ln in cp.stdout.splitlines() if ln.startswith("RESULT ")]
        if not line:
            print(f"{j['name']}: FAILED\n{cp.stderr[-2000:]}")
            continue
        r = json.loads(line[0][7:])
        results.append(r)
        print(f"{r['name']:38s} params {r.get('params_M', float('nan')):8.3f}M size {r.get('size_MB', float('nan')):8.2f}MB "
              f"lat {r['latency_ms_median']:9.1f}ms peakRSS {r['peak_rss_MB']:7.0f}MB "
              f"{'gpu ' + format(r['gpu_peak_MB'], '.0f') + 'MB' if 'gpu_peak_MB' in r else ''}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"size": a.size, "results": results}, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
