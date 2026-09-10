#!/usr/bin/env python3
"""Compute Pi0.5 action stats in GR00T-style relative EEF space."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation

def rel(target, current):
    rc = Rotation.from_rotvec(current[3:6]).as_matrix()
    rt = Rotation.from_rotvec(target[3:6]).as_matrix()
    return np.r_[rc.T @ (target[:3] - current[:3]), Rotation.from_matrix(rc.T @ rt).as_rotvec()]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset-root", type=Path, required=True); ap.add_argument("--chunk-size", type=int, default=50); ap.add_argument("--output", type=Path)
    a = ap.parse_args(); root = a.dataset_root
    df = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").glob("*/*.parquet"))], ignore_index=True)
    state = np.stack(df["observation.state"]).astype(np.float64); action = np.stack(df["action"]).astype(np.float64); ep = df["episode_index"].to_numpy()
    rows = []
    for i in range(len(df) - a.chunk_size + 1):
        if ep[i] != ep[i + a.chunk_size - 1]: continue
        for target in action[i:i + a.chunk_size]:
            row = target.copy(); row[:6] = rel(target[:6], state[i, 18:24]); row[6:12] = rel(target[6:12], state[i, 24:30]); rows.append(row)
    x = np.asarray(rows, dtype=np.float32)
    payload = json.loads((root / "meta" / "stats.json").read_text())
    payload["action"] = {k: v.tolist() for k, v in {"count": np.full(14, len(x), dtype=np.int64), "mean": x.mean(0), "std": x.std(0), "min": x.min(0), "max": x.max(0), "q01": np.percentile(x, 1, 0), "q10": np.percentile(x, 10, 0), "q50": np.percentile(x, 50, 0), "q90": np.percentile(x, 90, 0), "q99": np.percentile(x, 99, 0)}.items()}
    out = a.output or root / "meta" / "stats.json"; out.write_text(json.dumps(payload, indent=4) + "\n"); print(f"wrote {out}: {len(x)} rows")
if __name__ == "__main__": main()
