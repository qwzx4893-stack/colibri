"""Hardware autoprofiler: measure this machine, do not trust its name.

Runs the native probes (`lma_hwprof`, `lma_hwprof_cuda`, `iobench`) and writes one
versioned JSON profile. The scheduler cost model and the digital twin read the
numbers from here; nothing downstream carries a bandwidth constant.

    python -m lmatomic.hwprof --out hardware_profile.json [--disk-dir D] [--disk-gb 8]

A profile is bound to a fingerprint (CPU, RAM, GPU, driver, OS build, disk model,
tool version). `is_current(profile)` says whether it still describes this machine;
the Model Agent rebuilds a stale one instead of reusing it.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = 1
HERE = Path(__file__).resolve().parent
CDIR = HERE.parent / "c"
EXE = ".exe" if sys.platform == "win32" else ""


def _run(cmd, timeout=600, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env,
                       encoding="utf-8", errors="replace")
    return p.returncode, p.stdout, p.stderr


def _ps(script):
    rc, out, _ = _run(["powershell", "-NoProfile", "-Command", script], timeout=60)
    return out.strip() if rc == 0 else ""


def fingerprint():
    fp = {"os": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    if sys.platform == "win32":
        fp["cpu"] = _ps("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name")
        fp["cores_physical"] = int(_ps("(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum") or 0)
        fp["cores_logical"] = os.cpu_count()
        fp["ram_gb"] = round(int(_ps("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory") or 0) / 2**30, 1)
        fp["disk"] = _ps("(Get-PhysicalDisk | Select-Object -First 1).FriendlyName")
    else:
        fp["cpu"] = platform.processor()
        fp["cores_logical"] = os.cpu_count()
    rc, out, _ = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], timeout=30)
    fp["gpu"] = out.strip().splitlines()[0] if rc == 0 and out.strip() else None
    fp["tool_version"] = SCHEMA
    return fp


def fingerprint_id(fp):
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]


def _json_out(exe, args=(), timeout=600):
    path = CDIR / (exe + EXE)
    if not path.exists():
        return {"error": f"{path.name} is not built (make -C c {exe}{EXE})"}
    env = dict(os.environ)
    cuda_bin = os.environ.get("CUDA_PATH")
    if cuda_bin:
        env["PATH"] = str(Path(cuda_bin) / "bin" / "x64") + os.pathsep + str(Path(cuda_bin) / "bin") + os.pathsep + env["PATH"]
    rc, out, err = _run([str(path), *map(str, args)], timeout=timeout, env=env)
    if rc != 0:
        return {"error": (err or out).strip()[:400]}
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        return {"error": f"bad JSON from {exe}: {e}", "raw": out[:400]}


IOB_RE = re.compile(r"-> ([0-9.]+) GB/s \(([0-9.]+) effective ms/block\)")


def _make_disk_file(path: Path, gb: float):
    """Random incompressible data, real blocks (a sparse file would read as zeros)."""
    want = int(gb * 2**30)
    if path.exists() and path.stat().st_size >= want:
        return
    chunk = os.urandom(16 << 20)
    with open(path, "wb") as f:
        written = 0
        while written < want:
            f.write(chunk); written += len(chunk)
            chunk = chunk[1:] + chunk[:1]          # cheap per-block variation, still incompressible
        f.flush(); os.fsync(f.fileno())


def disk_profile(disk_dir: Path, gb: float, blocks_mb=(0.5, 3.5, 6.3, 13, 25, 64), threads=(1, 2, 4, 8)):
    iob = CDIR / ("iobench" + EXE)
    if not iob.exists():
        return {"error": f"{iob.name} is not built (make -C c iobench)"}
    disk_dir.mkdir(parents=True, exist_ok=True)
    f = disk_dir / "lma_hwprof_probe.bin"
    _make_disk_file(f, gb)
    rows = []
    for b in blocks_mb:
        for t in threads:
            n = max(32, int(min(6.0, 3.0 * t) * 1024 / b / 2))       # ~2-3 GB per point
            rc, out, err = _run([str(iob), str(f), str(b), str(n), str(t), "1"], timeout=300)
            m = IOB_RE.search(out)
            if rc == 0 and m:
                rows.append({"block_mb": b, "threads": t, "gbs": float(m.group(1)), "ms_per_block": float(m.group(2)),
                             "direct": "NO_BUFFERING" not in err and "O_DIRECT is unavailable" not in err})
    try:
        f.unlink()
    except OSError:
        pass
    return {"file_gb": gb, "unbuffered": True, "points": rows}


def build_profile(disk_dir: Path, disk_gb: float, threads=None):
    t0 = time.time()
    fp = fingerprint()
    args = ["--threads", threads] if threads else []
    cpu = _json_out("lma_hwprof", args)
    gpu = _json_out("lma_hwprof_cuda")
    disk = disk_profile(disk_dir, disk_gb)
    prof = {
        "schema": SCHEMA,
        "kind": "hardware_profile",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": fp, "fingerprint_id": fingerprint_id(fp),
        "measured": {"cpu_ram": cpu, "gpu": gpu, "nvme": disk},
        "measurement_seconds": round(time.time() - t0, 1),
    }
    prof["cost_model"] = derive_cost_model(prof["measured"])
    return prof


def nvme_gbs(disk, block_mb, threads):
    """Bandwidth (GB/s) for a read of `block_mb` at `threads` in flight, interpolated
    on the measured grid (log block size, nearest thread count at or below)."""
    pts = [p for p in disk.get("points", []) if p["threads"] <= threads] or disk.get("points", [])
    if not pts:
        return None
    tmax = max(p["threads"] for p in pts)
    pts = [p for p in pts if p["threads"] == tmax]
    pts.sort(key=lambda p: p["block_mb"])
    import math
    if block_mb <= pts[0]["block_mb"]:
        return pts[0]["gbs"]
    for a, b in zip(pts, pts[1:]):
        if block_mb <= b["block_mb"]:
            w = (math.log(block_mb) - math.log(a["block_mb"])) / (math.log(b["block_mb"]) - math.log(a["block_mb"]))
            return a["gbs"] + w * (b["gbs"] - a["gbs"])
    return pts[-1]["gbs"]


def derive_cost_model(m):
    """Flat parameters the scheduler and twin consume (units: bytes, seconds)."""
    cm = {}
    disk = m.get("nvme", {})
    if disk.get("points"):
        best = max(disk["points"], key=lambda p: p["gbs"])
        cm["nvme_peak_gbs"] = best["gbs"]
        cm["nvme_peak_at"] = {"block_mb": best["block_mb"], "threads": best["threads"]}
        cm["nvme_gbs_by_block_mb_qd8"] = {str(p["block_mb"]): p["gbs"] for p in disk["points"] if p["threads"] == 8}
        cm["nvme_gbs_by_block_mb_qd1"] = {str(p["block_mb"]): p["gbs"] for p in disk["points"] if p["threads"] == 1}
    c = m.get("cpu_ram", {})
    if "ram_read_gbs_all" in c:
        cm["ram_gbs"] = c["ram_read_gbs_all"]
        cm["cpu_expert_gbs_all_threads"] = c["cpu_expert_weight_gbs_all_threads"]
        cm["cpu_expert_gbs_1thread_resident"] = c["cpu_expert_weight_gbs_1thread_resident"]
    g = m.get("gpu", {})
    if "transfer" in g:
        big = g["transfer"][-2]
        cm["h2d_pinned_gbs"] = big["h2d_pinned_gbs"]
        cm["h2d_pageable_gbs"] = big["h2d_pageable_gbs"]
        cm["d2h_pinned_gbs"] = big["d2h_pinned_gbs"]
        cm["vram_read_gbs"] = g["vram_read_gbs"]
        cm["gpu_expert_ms"] = g["gpu_expert_ms_by_tokens"]
        cm["copy_kernel_overlap_fraction"] = g["overlap"]["overlap_fraction"]
        cm["pinned_alloc_gb"] = g.get("pinned_alloc_gb")
        cm["vram_free_gb"] = g["vram_free_gb"]
    return cm


def is_current(profile) -> bool:
    return profile.get("schema") == SCHEMA and profile.get("fingerprint_id") == fingerprint_id(fingerprint())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="hardware_profile.json")
    ap.add_argument("--disk-dir", default=str(Path.home() / ".lmatomic" / "probe"))
    ap.add_argument("--disk-gb", type=float, default=8.0)
    ap.add_argument("--threads", default=None)
    a = ap.parse_args(argv)
    prof = build_profile(Path(a.disk_dir), a.disk_gb, a.threads)
    Path(a.out).write_text(json.dumps(prof, indent=2), encoding="utf-8")
    print(json.dumps(prof["cost_model"], indent=2))
    print(f"wrote {a.out} ({prof['measurement_seconds']} s)")


if __name__ == "__main__":
    main()
