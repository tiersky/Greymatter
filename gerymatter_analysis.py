#!/usr/bin/env python
# ---------------------------------------------------------------------------
#  USAGE:  python greymatter_etl.py  path/to/influencers.csv
#  OUTPUT: influencer_modelling_ready.parquet
# ---------------------------------------------------------------------------

import sys, re, pathlib, numpy as np, pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
import chardet          # pip install chardet
# ---------------------------------------------------------------------------
csv_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("influencers.csv")

# 1️⃣  Load CSV with best-guess encoding -------------------------------------
def read_robust(path: pathlib.Path) -> pd.DataFrame:
    with path.open("rb") as f:
        enc = chardet.detect(f.read(50_000))["encoding"] or "windows-1252"
    try:
        return pd.read_csv(path, encoding=enc, engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="windows-1252", engine="python")

raw = read_robust(csv_path)

# 2️⃣  Normalise headers (dots, spaces, ampersands) --------------------------
raw = raw.rename(columns=lambda c: (c.strip()
                                      .replace('.', '')
                                      .replace('&', 'and')
                                      .replace(' ', '_')))

print("✅  Columns:", list(raw.columns)[:15], "…")

# 3️⃣  Ensure numeric Est_Post_Price (already averaged by you) ---------------
price_cols = [c for c in raw.columns if 'post' in c.lower() and 'price' in c.lower()]
if price_cols:
    price_col = price_cols[0]
    raw[price_col] = pd.to_numeric(raw[price_col], errors="coerce")
else:
    print("⚠️  No price column found")

# 4️⃣  Categories → list -----------------------------------------------------
#      (force to string first, then split)
if "Category" in raw.columns:
    raw["Category_list"] = (
        raw["Category"]
            .fillna('')
            .astype(str)
            .str.split(',')
            .apply(lambda lst: [c.strip() for c in lst if c.strip()])
    )
else:
    print("⚠️  Category column not found, creating empty list")
    raw["Category_list"] = [[] for _ in range(len(raw))]

# 5️⃣  Interests → % columns -------------------------------------------------
pair_pat = re.compile(r"(?P<name>[^:]+):\s*(?P<pct>\d+)%")
def parse_interest(cell) -> dict:
    cell = str(cell) if not pd.isna(cell) else ''
    return {m["name"].strip(): int(m["pct"]) for m in pair_pat.finditer(cell)}

if "Interests" in raw.columns:
    raw["Interest_dict"] = raw["Interests"].apply(parse_interest)
    all_ints = sorted({k for d in raw["Interest_dict"] for k in d})
    for k in all_ints:
        raw[f"int_{k}"] = raw["Interest_dict"].apply(lambda d: d.get(k, 0)).astype("uint8")
else:
    print("⚠️  Interests column not found, skipping interest parsing")
    raw["Interest_dict"] = [{} for _ in range(len(raw))]

# 6️⃣  Category one-hot ------------------------------------------------------
mlb = MultiLabelBinarizer(sparse_output=True)
cat_ohe = mlb.fit_transform(raw["Category_list"])
cat_df  = pd.DataFrame.sparse.from_spmatrix(
              cat_ohe, index=raw.index,
              columns=[f"cat_{c}" for c in mlb.classes_]
          )
raw = pd.concat([raw, cat_df], axis=1)

# 7️⃣  Dimensionality reduction (SVD) ----------------------------------------
cat_cols = [c for c in raw if c.startswith("cat_")]
int_cols = [c for c in raw if c.startswith("int_")]

transformers = []
reduced_cols = []

if cat_cols:
    svd_cat = TruncatedSVD(n_components=min(10, len(cat_cols)), random_state=42)
    transformers.append(("cat", svd_cat, cat_cols))
    reduced_cols.extend([f"catSVD_{i}" for i in range(min(10, len(cat_cols)))])

if int_cols:
    svd_int = TruncatedSVD(n_components=min(25, len(int_cols)), random_state=42)
    transformers.append(("int", svd_int, int_cols))
    reduced_cols.extend([f"intSVD_{i}" for i in range(min(25, len(int_cols)))])

if transformers:
    ct = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
    X_reduced = ct.fit_transform(raw)
    X_reduced = pd.DataFrame(X_reduced, columns=reduced_cols, index=raw.index)
else:
    print("⚠️  No categorical or interest columns found for SVD")
    X_reduced = pd.DataFrame(index=raw.index)

# 8️⃣  Assemble modelling frame ---------------------------------------------
numeric_keep = [c for c in raw.columns
                if raw[c].dtype.kind in "fiu" and
                   c not in (*cat_cols, *int_cols)]
model_df = pd.concat([raw[numeric_keep], X_reduced], axis=1)

print("✅  Final shape:", model_df.shape)

# 9️⃣  Save ------------------------------------------------------------------
model_df.to_parquet("influencer_modelling_ready.parquet")
print("🚀  Saved → influencer_modelling_ready.parquet")
