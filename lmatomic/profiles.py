"""Where a model's measured profile lives and how the runtime consumes it.

Profiles are stored in the user's data directory, keyed by the model's identity hash, never inside the model directory
(original model files are never touched). `runtime_env()` turns a valid profile into engine environment defaults; the
launcher applies them with setdefault, so anything the user sets explicitly still wins.
"""
import json
import os
import sys
from pathlib import Path

from . import agent, hwprof


def user_data_dir():
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~\\AppData\\Local"))
    else:
        base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return Path(base) / "lmatomic"


def profile_dir(model_dir):
    return user_data_dir() / "agents" / agent.model_hash(Path(model_dir))


def load_profile(model_dir):
    """The stored profile if it still matches this model AND this hardware, else None (a stale profile is never used)."""
    d = profile_dir(model_dir)
    p, hwp = d / "agent_profile.json", d / "hardware_profile.json"
    try:
        prof = json.loads(p.read_text(encoding="utf-8"))
        hw = json.loads(hwp.read_text(encoding="utf-8"))
    except Exception:
        return None
    if prof.get("schema") != agent.SCHEMA or prof.get("model_hash") != agent.model_hash(Path(model_dir)):
        return None
    if prof.get("hardware_fingerprint_id") != hw.get("fingerprint_id") or not hwprof.is_current(hw):
        return None
    return prof


def runtime_env(profile):
    """Engine environment implied by a profile: the circuit-breaker baseline and the selected High-Fidelity profile."""
    env = {}
    pol = ((profile or {}).get("strategy") or {}).get("performance_policy") or {}
    if pol.get("cpu_baseline_tok_s"):
        env["LMA_CPU_BASELINE_TOK_S"] = str(pol["cpu_baseline_tok_s"])
    sel = pol.get("selected_profile")
    if pol.get("mode") == "high_fidelity" and sel:
        env.update(sel.get("env", {}))
    return env
