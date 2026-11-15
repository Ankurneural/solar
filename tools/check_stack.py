"""Exit 0 if <run>/stack.yaml's OOF PQ beats a threshold (gate a queued upload on it).

  python tools/check_stack.py runs/w9_all 0.4647
"""
import sys

import yaml

run, best = sys.argv[1], float(sys.argv[2])
q = yaml.safe_load(open(f"{run}/stack.yaml"))["oof_pq"]
print(f"{run} stack OOF {q:.4f} vs {best:.4f}")
sys.exit(0 if q > best else 1)
