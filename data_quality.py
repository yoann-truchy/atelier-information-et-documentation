import pandas as pd
import numpy as np
import json
import sys


CSV_PATH = "student_lifestyle_100k.csv"  # <-- adapte le chemin
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]




print("=== APERÇU ===")
print(f"Lignes: {df.shape[0]} | Colonnes: {df.shape[1]}")
print("\nTypes de colonnes:")
print(df.dtypes)

# -----------------------
# 1) Stats descriptives
# -----------------------
print("\n=== STATS DESCRIPTIVES (pandas) ===")
desc = df.describe(include="all").T
print(desc)

# -----------------------
# 2) Valeurs manquantes
# -----------------------
print("\n=== VALEURS MANQUANTES ===")
missing_count = df.isna().sum()
missing_pct = (df.isna().mean() * 100).round(3)

print("\nManquants par colonne (count):")
print(missing_count[missing_count > 0].sort_values(ascending=False))

print("\nManquants par colonne (%):")
print(missing_pct[missing_pct > 0].sort_values(ascending=False))

total_cells = df.size
missing_cells = int(df.isna().sum().sum())
missing_pct_global = (missing_cells / total_cells * 100) if total_cells else 0.0

print(f"\nManquants GLOBAL: {missing_cells}/{total_cells} cellules = {missing_pct_global:.3f}%")

# -----------------------
# 3) Doublons
# -----------------------
print("\n=== DOUBLONS ===")
dup_rows = int(df.duplicated().sum())
print(f"Doublons de lignes: {dup_rows}")

if "Student_ID" in df.columns:
    dup_student_id = int(df["Student_ID"].duplicated().sum())
    print(f"Doublons Student_ID: {dup_student_id}")

# -----------------------
# 4) Valeurs aberrantes (global IQR)
# -----------------------
print("\n=== VALEURS ABERRANTES (IQR) — GLOBAL ===")
num = df.select_dtypes(include=[np.number])

if num.shape[1] == 0:
    print("Aucune colonne numérique -> pas de détection d'outliers.")
else:
    q1 = num.quantile(0.25)
    q3 = num.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (num.lt(lower, axis=1)) | (num.gt(upper, axis=1))

    outlier_cells = int(outlier_mask.sum().sum())
    rows_with_outlier = int(outlier_mask.any(axis=1).sum())

    total_numeric_cells = int(num.size)
    pct_outlier_cells = (outlier_cells / total_numeric_cells * 100) if total_numeric_cells else 0.0
    pct_rows_outlier = (rows_with_outlier / len(df) * 100) if len(df) else 0.0

    print(f"Colonnes numériques: {num.shape[1]}")
    print(f"Cellules numériques totales: {total_numeric_cells}")
    print(f"Cellules outliers (IQR): {outlier_cells} ({pct_outlier_cells:.3f}%)")
    print(f"Lignes avec ≥1 outlier: {rows_with_outlier} ({pct_rows_outlier:.3f}%)")

    # Optionnel : top colonnes qui ont le plus d'outliers
    outliers_per_col = outlier_mask.sum().sort_values(ascending=False)
    print("\nTop colonnes avec le plus d'outliers (count):")
    print(outliers_per_col.head(10))

# -----------------------
# 5) Matrice de correlation (Pearson et Spearman)
# -----------------------

# Garder uniquement les colonnes numériques
num_df = df.select_dtypes(include=[np.number])
for corr in ["pearson","spearman"]:

    corr_matrix = num_df.corr(method=corr).round(4)

    matrix_str = np.array2string(
        corr_matrix.to_numpy(),
        separator=",",
        max_line_width=10**9
    )

    print(matrix_str)
