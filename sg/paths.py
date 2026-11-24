"""Where things are. The trainers are submodules (pinned): rl-v3/ (PPO, the net) and bv-v1/ (the rollout teacher,
labels, matches). Python is rl-v3's venv (SG_PY overrides): the sibling checkout's, as bv-v1 does."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL3 = os.path.join(ROOT, "rl-v3")
BV = os.path.join(ROOT, "bv-v1")
MANIFESTS = os.path.join(ROOT, "manifests")
RESULTS = os.path.join(ROOT, "results")
EXCLUSIONS = os.path.join(MANIFESTS, "exclusions.json")
PY = os.environ.get("SG_PY") or os.path.abspath(os.path.join(ROOT, "..", "rl-v3", ".venv", "bin", "python"))
C2 = os.path.join(RL3, "opponents", "rl3_c2_2654.npy")
C1 = os.path.join(RL3, "opponents", "rl3_c1_2300.npy")
RL1 = os.path.join(RL3, "opponents", "rl_r2_2200.npy")


def use_bv():
    """Import bv-v1's package (bv.core builds bv.so inside the submodule)."""
    if BV not in sys.path:
        sys.path.insert(0, BV)


def use_rl3():
    if RL3 not in sys.path:
        sys.path.insert(0, RL3)
