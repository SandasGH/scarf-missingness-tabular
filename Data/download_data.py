import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "Raw")

# ── 1. UCI Credit Card Default ────────────────────────────────────────────────
uci_xls  = os.path.join(RAW_DIR, "default of credit card clients.xls")
uci_csv  = os.path.join(RAW_DIR, "default_credit_card.csv")

print("=" * 60)
print("DATASET 1: UCI Credit Card Default")
print("=" * 60)

if not os.path.exists(uci_csv):
    df_uci = pd.read_excel(uci_xls, header=1)   # row 0 is a title row
    df_uci.to_csv(uci_csv, index=False)
    print(f"Saved clean CSV -> {uci_csv}")
else:
    df_uci = pd.read_csv(uci_csv)
    print(f"Loaded existing CSV from {uci_csv}")

print(f"\nShape : {df_uci.shape[0]} rows × {df_uci.shape[1]} columns")

print("\nFirst 5 rows:")
print(df_uci.head().to_string())

print("\nMissing values per column:")
missing_uci = df_uci.isnull().sum()
print(missing_uci[missing_uci > 0].to_string() if missing_uci.sum() > 0 else "  None")

# Target column is 'default payment next month' (or PAY_AMT after renaming)
target_col_uci = "default payment next month"
if target_col_uci in df_uci.columns:
    counts = df_uci[target_col_uci].value_counts().sort_index()
    print(f"\nClass balance  ({target_col_uci}):")
    for label, n in counts.items():
        print(f"  {label}: {n:,}  ({n / len(df_uci) * 100:.1f}%)")
else:
    print(f"\nTarget column '{target_col_uci}' not found. Columns: {list(df_uci.columns)}")

# ── 2. Telco Customer Churn ───────────────────────────────────────────────────
telco_csv = os.path.join(RAW_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("\n" + "=" * 60)
print("DATASET 2: Telco Customer Churn")
print("=" * 60)

df_telco = pd.read_csv(telco_csv)
print(f"\nShape : {df_telco.shape[0]} rows × {df_telco.shape[1]} columns")

print("\nFirst 5 rows:")
print(df_telco.head().to_string())

print("\nMissing values per column:")
# TotalCharges may have blank strings that pandas doesn't auto-detect
df_telco["TotalCharges"] = pd.to_numeric(df_telco["TotalCharges"], errors="coerce")
missing_telco = df_telco.isnull().sum()
print(missing_telco[missing_telco > 0].to_string() if missing_telco.sum() > 0 else "  None")

target_col_telco = "Churn"
counts = df_telco[target_col_telco].value_counts().sort_index()
print(f"\nClass balance  ({target_col_telco}):")
for label, n in counts.items():
    print(f"  {label}: {n:,}  ({n / len(df_telco) * 100:.1f}%)")

print("\nDone.")
