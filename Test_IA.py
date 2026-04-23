import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys


opt = ["x","x","x"]
for i in range(len(sys.argv)):
    opt[i] = sys.argv[i]

df = pd.read_csv("student_lifestyle_100k.csv")


if opt[1] == "corr":

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Conversion de Gender
    df["Gender"] = df["Gender"].map({
        "Male": 1,
        "Female": 0
    })

    # Conversion de Depression
    df["Depression"] = df["Depression"].map({
        True: 1,
        False: 0
    })


    # Label encoding (pas ohe)
    le = LabelEncoder()
    df["Department"] = le.fit_transform(df["Department"])


    df_numeric = df.select_dtypes(include=["int64", "float64"])


    corr_matrix = df_numeric.corr()
    #print(corr_matrix)

    if opt[2] == "m":
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matrice de corrélation")
        plt.show()

    else : 
        # Top 3 des variables les + correlé avec depression (en VA)
        corr_depression = corr_matrix["Depression"].drop("Depression")
        top_3 = corr_depression.abs().sort_values(ascending=False).head(3)
        print("Top 3 variables les plus corrélées avec Depression :")
        print(top_3)

else:

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib

    from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score

    # --- Ton setup (df doit déjà être chargé) ---
    TARGET = "Depression"
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].astype(int)

    if "Student_ID" in X.columns:
        X = X.drop(columns=["Student_ID"])

    # sécurité: CGPA doit exister et être numérique
    assert "CGPA" in X.columns, "Colonne 'CGPA' introuvable dans X."
    X["CGPA"] = pd.to_numeric(X["CGPA"], errors="coerce")
    if X["CGPA"].isna().any():
        # si tu préfères, tu peux drop les lignes au lieu d'imputer
        X["CGPA"] = X["CGPA"].fillna(X["CGPA"].median())

    # Charger le pipeline (préprocess + modèle)
    model = joblib.load("depression_classifier.joblib")

    # --- Expérience bruit ---
    noise_levels = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20]  # 0%, 1%, 2%, 5%, 10%, 20%
    rng = np.random.default_rng(42)  # reproductible

    recalls, precisions, f1s, pr_aucs = [], [], [], []

    for nl in noise_levels:
        X_noisy = X.copy()

        # bruit multiplicatif: CGPA' = CGPA * (1 + epsilon), epsilon ~ N(0, nl)
        if nl > 0:
            eps = rng.normal(loc=0.0, scale=nl, size=len(X_noisy))
            X_noisy["CGPA"] = X_noisy["CGPA"] * (1.0 + eps)

        # Predictions (classes)
        y_pred = model.predict(X_noisy)

        # Scores pour PR-AUC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_noisy)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_noisy)
        else:
            raise ValueError("Impossible de calculer PR-AUC: pas de predict_proba ni decision_function.")

        # Métriques
        recalls.append(recall_score(y, y_pred))
        precisions.append(precision_score(y, y_pred, zero_division=0))
        f1s.append(f1_score(y, y_pred, zero_division=0))
        pr_aucs.append(average_precision_score(y, y_score))

    # --- Affichage tableau console ---
    print("Bruit(%) | Recall  | Precision | F1     | PR-AUC")
    print("-" * 50)
    for nl, r, p, f, a in zip(noise_levels, recalls, precisions, f1s, pr_aucs):
        print(f"{int(nl*100):>7d} | {r:>6.4f} | {p:>9.4f} | {f:>6.4f} | {a:>6.4f}")

    # --- Courbe d'évolution ---
    x_pct = [nl * 100 for nl in noise_levels]

    plt.figure(figsize=(9, 5))
    plt.plot(x_pct, recalls, marker="o", label="Recall")
    plt.plot(x_pct, precisions, marker="o", label="Precision")
    plt.plot(x_pct, f1s, marker="o", label="F1-score")
    plt.plot(x_pct, pr_aucs, marker="o", label="PR-AUC")

    plt.xlabel("Bruit appliqué sur CGPA (%)")
    plt.ylabel("Score")
    plt.title("Impact du bruit sur CGPA sur les métriques du modèle")
    plt.xticks(x_pct)
    plt.ylim(0, 1)
    plt.grid(True, linewidth=0.5)
    plt.legend()
    plt.show()





