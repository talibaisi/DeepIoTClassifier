#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:57:44 2026

@author: thaer

Step 1: draw a stratified 50,000-flow subset from the CICIoT2023 CSV release
(one sub-folder per class). Each class keeps its share of the full release, with
a minimum of 10 flows per class so that every class can appear in both partitions.
Resumable: each class is written to data/parts/<class>.csv.
"""
import os, glob, json, sys
import numpy as np, pandas as pd

SRC = [a for a in sys.argv[1:] if not a.startswith("--")][0]
N_TARGET, MIN_PER_CLASS, SEED = 50_000, 10, 42
BALANCED = "--balanced" in sys.argv   # 25,000 benign + 25,000 attack (attacks proportional across the 33 classes)
PARTS = "data/parts_bal" if BALANCED else "data/parts"
OUTF = "data/ciciot2023_balanced_50k.csv" if BALANCED else "data/ciciot2023_subset_50k.csv"
os.makedirs(PARTS, exist_ok=True)

def count_rows(f):
    with open(f, "rb") as fh:
        return sum(1 for _ in fh) - 1

files = {}
for d in sorted(os.listdir(SRC)):
    p = os.path.join(SRC, d)
    if os.path.isdir(p):
        label = "BenignTraffic" if d.startswith("Benign") else d
        files[label] = [(f, count_rows(f)) for f in sorted(glob.glob(os.path.join(p, "*.csv")))]
class_rows = {c: sum(n for _, n in v) for c, v in files.items()}
total = sum(class_rows.values())
if BALANCED:
    att_total = total - class_rows["BenignTraffic"]
    target = {c: max(MIN_PER_CLASS, int(round(N_TARGET / 2 * n / att_total))) for c, n in class_rows.items() if c != "BenignTraffic"}
    target["BenignTraffic"] = N_TARGET // 2
else:
    target = {c: max(MIN_PER_CLASS, int(round(N_TARGET * n / total))) for c, n in class_rows.items()}
diff = sum(target.values()) - N_TARGET          # fix rounding on the largest classes only
big = sorted([c for c in target if target[c] > 10 * MIN_PER_CLASS and c != "BenignTraffic"], key=lambda c: -target[c])
i = 0
while diff != 0:
    s = int(np.sign(diff)); target[big[i % len(big)]] -= s; diff -= s; i += 1
assert sum(target.values()) == N_TARGET and min(target.values()) >= MIN_PER_CLASS

for ci, (c, flist) in enumerate(sorted(files.items())):
    out = f"{PARTS}/{c}.csv"
    if os.path.exists(out):
        continue
    rng = np.random.default_rng(SEED + ci)
    need, n_c = target[c], class_rows[c]
    # draw 3x the needed number of row positions (margin for NaN/inf rows and duplicates)
    k_all = min(n_c, max(need * 3, need + 20))
    picks = np.sort(rng.choice(n_c, size=k_all, replace=False))
    rows, offset, header = [], 0, None
    for f, n in flist:
        sel = set((picks[(picks >= offset) & (picks < offset + n)] - offset).tolist())
        if sel:
            with open(f, "r") as fh:
                header = fh.readline()
                for i, line in enumerate(fh):
                    if i in sel:
                        rows.append(line)
        offset += n
    from io import StringIO
    df = pd.read_csv(StringIO(header + "".join(rows)))
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    df = df.sample(n=min(need, len(df)), random_state=SEED)
    df["label"] = c
    df.to_csv(out, index=False)
    print(f"{c:28s} release={n_c:>9,d}  target={need:>5d}  sampled={len(df):>5d}", flush=True)

parts = [pd.read_csv(f"{PARTS}/{c}.csv") for c in sorted(files)]
data = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
data.to_csv(OUTF, index=False)
json.dump({"release_rows_per_class": class_rows, "release_total": total, "target_per_class": target,
           "subset_rows_per_class": data["label"].value_counts().to_dict(),
           "subset_total": int(len(data)), "n_features": int(data.shape[1] - 1),
           "features": [c for c in data.columns if c != "label"]},
          open("results/01_subset_summary%s.json" % ("_balanced" if BALANCED else ""), "w"), indent=2)
print("subset:", data.shape)
