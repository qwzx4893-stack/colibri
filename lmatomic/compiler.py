"""Atomic Compiler: turns a model's experts into the compact int4 runtime format, streaming and resumable.

  * Streaming: one shard = a few layers; the engine backend (`QCOMPILE`) holds a single expert in memory at a time,
    so peak RAM does not grow with model size.
  * Checkpointed and resumable: a shard is written to `<name>.tmp`, validated, then atomically renamed; the manifest
    (model identity, quantiser settings, per-shard size + sha256) is rewritten atomically after every shard. Re-running
    skips every shard whose file still matches the manifest, so an interrupted compile continues where it stopped.
  * Non-destructive: the source model directory is only read. Output goes to a separate overlay directory (default: the
    user's data dir); the runtime reads it through LMA_EXPERT_OVERLAY and the original int8 experts stay the CPU fallback.
"""
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

from . import agent, profiles

FORMAT = "int4-g32-fp16"
MANIFEST = "manifest.json"


def default_out(model_dir):
    return profiles.profile_dir(model_dir) / "int4"


def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _atomic_json(path, obj):
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def compile_int4(model_dir, out_dir=None, layers_per_shard=2, clip=True, exe=None, log=print, timeout=7200):
    model_dir = Path(model_dir)
    out_dir = Path(out_dir) if out_dir else default_out(model_dir)
    if out_dir.resolve() == model_dir.resolve() or model_dir.resolve() in out_dir.resolve().parents:
        raise ValueError("refusing to write the overlay into the source model directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = exe or str(Path(__file__).resolve().parent.parent / "c" / ("olmoe.exe" if sys.platform == "win32" else "olmoe"))
    graph = agent.inspect_model(model_dir)
    n_layers, n_exp = graph["moe_layers"], graph["experts_per_layer"]
    mpath = out_dir / MANIFEST
    ident = {"format": FORMAT, "model_hash": agent.model_hash(model_dir), "clip": bool(clip), "layers": n_layers,
             "experts_per_layer": n_exp, "layers_per_shard": layers_per_shard}
    man = {"schema": 1, **ident, "shards": {}, "complete": False}
    if mpath.exists():
        old = json.loads(mpath.read_text(encoding="utf-8"))
        if all(old.get(k) == v for k, v in ident.items()):
            man = old
        else:
            log("[compile] stored manifest belongs to a different model/settings: starting a fresh overlay")
            for f in out_dir.glob("experts-int4-*.safetensors*"):
                f.unlink()
    man["complete"] = False
    done = skipped = 0
    for lo in range(0, n_layers, layers_per_shard):
        hi = min(lo + layers_per_shard, n_layers) - 1
        name = f"experts-int4-L{lo:03d}-{hi:03d}.safetensors"
        path = out_dir / name
        ent = man["shards"].get(name)
        if ent and path.exists() and path.stat().st_size == ent["bytes"] and _sha256(path) == ent["sha256"]:
            skipped += 1
            log(f"[compile] {name}: already done (verified), skipping")
            continue
        tmp = out_dir / (name + ".tmp")
        env = dict(os.environ, SNAP=str(model_dir), QCOMPILE=f"{lo}-{hi}", QC_OUT=str(tmp), LMA_HF_INT4_CLIP="1" if clip else "0")
        r = subprocess.run([exe], env=env, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 or not tmp.exists():
            raise RuntimeError(f"compile of layers {lo}-{hi} failed: {r.stderr[-400:] or r.stdout[-400:]}")
        hdr, _ = _header(tmp)
        expect = 2 * (hi - lo + 1) * n_exp
        if len([k for k in hdr if k != "__metadata__"]) != expect:
            raise RuntimeError(f"{name}: expected {expect} tensors, header has {len(hdr) - 1}")
        os.replace(tmp, path)
        man["shards"][name] = {"layers": [lo, hi], "bytes": path.stat().st_size, "sha256": _sha256(path)}
        _atomic_json(mpath, man)                 # checkpoint after every shard
        done += 1
        log(f"[compile] {name}: layers {lo}-{hi}, {man['shards'][name]['bytes'] / 1e6:.0f} MB")
    man["complete"] = True
    man["total_bytes"] = sum(s["bytes"] for s in man["shards"].values())
    _atomic_json(mpath, man)
    log(f"[compile] overlay complete: {done} shard(s) built, {skipped} reused, {man['total_bytes'] / 1e9:.2f} GB -> {out_dir}")
    return man


def overlay_for(model_dir, out_dir=None):
    """The overlay directory if a COMPLETE compile matching this exact model exists, else None."""
    out_dir = Path(out_dir) if out_dir else default_out(model_dir)
    try:
        man = json.loads((out_dir / MANIFEST).read_text(encoding="utf-8"))
    except Exception:
        return None
    if not man.get("complete") or man.get("format") != FORMAT or man.get("model_hash") != agent.model_hash(Path(model_dir)):
        return None
    if not all((out_dir / n).exists() and (out_dir / n).stat().st_size == s["bytes"] for n, s in man["shards"].items()):
        return None
    return str(out_dir)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="lmatomic compile")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--layers-per-shard", type=int, default=2)
    ap.add_argument("--no-clip", action="store_true")
    a = ap.parse_args(argv)
    compile_int4(a.model, a.out, a.layers_per_shard, clip=not a.no_clip)


if __name__ == "__main__":
    main()
