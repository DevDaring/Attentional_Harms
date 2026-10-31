import glob
import os

import pandas as pd

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
keys = []
for f in sorted(glob.glob("annotation/keys/*_key.csv")):
    if "withdrawn" in f:
        continue
    b = os.path.basename(f)[:-8]
    k = pd.read_csv(f, dtype=str)
    k["batch"] = b
    keys.append(k)
K = pd.concat(keys, ignore_index=True)
print(len(K), K.unit_id.nunique(), K.request_key.nunique())
dup = K[K.request_key.duplicated(keep=False)].sort_values("request_key")
print("dup request_key rows", len(dup))
print(dup[["batch", "sample", "arm", "condition", "family_id"]].head(20).to_string())
print(pd.crosstab([K["sample"], K["split"].fillna("-")], K["batch"]))
print(pd.crosstab([K["sample"], K["arm"]], K["condition"]))
print("families by sample")
print(K.groupby("sample").family_id.nunique())
print(K.groupby(["sample", "experiment"]).size())
