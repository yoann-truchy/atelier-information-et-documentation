import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "student_lifestyle_100k.csv"  # <-- adapte le chemin
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

print("=== APERÇU ===")
print(f"Lignes: {df.shape[0]} | Colonnes: {df.shape[1]}")
print("\nTypes de colonnes:")
print(df.dtypes)

# -----------------------
# 1) Stats descriptives -> BOXPLOTS
# -----------------------
print("\n=== STATS DESCRIPTIVES (GRAPHIQUES) ===")
num = df.select_dtypes(include=[np.number])

if num.shape[1] == 0:
    print("Aucune colonne numérique -> pas de boxplots.")
else:
    for col in num.columns:
        s = num[col].dropna().to_numpy()
        if s.size == 0:
            print(f"[SKIP] {col}: colonne vide (après dropna).")
            continue

        q1 = np.percentile(s, 25)
        med = np.percentile(s, 50)
        q3 = np.percentile(s, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = s[(s < lower) | (s > upper)]
        outlier_count = outliers.size

        plt.figure(figsize=(9, 2.2))
        plt.boxplot(
            s,
            vert=False,
            whis=1.5,
            showfliers=True,
            medianprops={"color": "red", "linewidth": 2.0},
            boxprops={"linewidth": 1.2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.7},
        )

        plt.title(f"Boxplot — {col}  |  outliers(IQR): {outlier_count}")
        plt.xlabel(col)

        # Annotation simple des quantiles (Q1 / Med / Q3)
        txt = f"Q1={q1:.3g} | Med={med:.3g} | Q3={q3:.3g}"
        plt.text(0.01, 0.90, txt, transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.show()

# -----------------------
# 2) Valeurs manquantes -> BAR CHART
# -----------------------
print("\n=== VALEURS MANQUANTES (GRAPHIQUES) ===")
missing_count = df.isna().sum()
missing_pct = (df.isna().mean() * 100)

total_cells = df.size
missing_cells = int(missing_count.sum())
missing_pct_global = (missing_cells / total_cells * 100) if total_cells else 0.0
print(f"Manquants GLOBAL: {missing_cells}/{total_cells} cellules = {missing_pct_global:.3f}%")

miss_cols = missing_pct[missing_pct > 0].sort_values(ascending=False)
if miss_cols.empty:
    print("Aucune valeur manquante détectée.")
else:
    plt.figure(figsize=(10, 4.5))
    plt.bar(miss_cols.index.astype(str), miss_cols.values)
    plt.title("Pourcentage de valeurs manquantes par colonne")
    plt.ylabel("% manquants")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.show()

# -----------------------
# 3) Doublons -> TEXTE (pas très utile en graph)
# -----------------------
print("\n=== DOUBLONS ===")
dup_rows = int(df.duplicated().sum())
print(f"Doublons de lignes: {dup_rows}")

if "Student_ID" in df.columns:
    dup_student_id = int(df["Student_ID"].duplicated().sum())
    print(f"Doublons Student_ID: {dup_student_id}")

# -----------------------
# 4) Valeurs aberrantes (IQR) — GLOBAL + BAR CHART par colonne
# -----------------------
print("\n=== VALEURS ABERRANTES (IQR) — GLOBAL ===")
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

    outliers_per_col = outlier_mask.sum().sort_values(ascending=False)
    print("\nTop colonnes avec le plus d'outliers (count):")
    print(outliers_per_col.head(10))

    # Graph: outliers par colonne (si au moins une colonne a des outliers)
    nonzero = outliers_per_col[outliers_per_col > 0]
    if not nonzero.empty:
        plt.figure(figsize=(10, 4.5))
        plt.bar(nonzero.index.astype(str), nonzero.values)
        plt.title("Nombre d'outliers (IQR) par colonne")
        plt.ylabel("Count outliers")
        plt.xticks(rotation=75, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print("Aucun outlier (IQR) détecté dans les colonnes numériques.")

# -----------------------
# 5) Corrélations -> HEATMAPS
# -----------------------
print("\n=== CORRÉLATIONS (HEATMAPS) ===")
num_df = df.select_dtypes(include=[np.number])

if num_df.shape[1] < 2:
    print("Pas assez de colonnes numériques pour calculer une corrélation.")
else:
    for method in ["pearson", "spearman"]:
        corr_matrix = num_df.corr(method=method).to_numpy()

        plt.figure(figsize=(8, 6))
        im = plt.imshow(corr_matrix, aspect="auto")
        plt.title(f"Matrice de corrélation ({method})")
        plt.colorbar(im)

        labels = num_df.columns.astype(str).to_list()
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)

        plt.tight_layout()
        plt.show()
