
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib


# ======================
# 1) Charger les données
# ======================
CSV_PATH = "student_lifestyle_100k.csv" 
df = pd.read_csv(CSV_PATH)

# ======================
# 2) Séparer X / y
# ======================
TARGET = "Depression"
assert TARGET in df.columns, f"Colonne cible '{TARGET}' introuvable."

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)  # False/True -> 0/1

#Student_ID n'apporte rien, on l'enlève
if "Student_ID" in X.columns:
    X = X.drop(columns=["Student_ID"])

# ======================
# 3) Gestion des colonnes numériques / catégorielles
# ======================
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# Encodage 
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
)


# ======================
# 4) Split train / test
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # important pour la gestion de classe classes déséquilibrées
)

# ======================
# 5) Modèles (choisis-en un)
# ======================

# A) Baseline solide (linéaire) + pondération auto pour classes déséquilibrées
logreg = LogisticRegression(
    max_iter=2000,
    class_weight="balanced", 
    n_jobs=None
)

# B) Modèle non-linéaire performant (arbres boosting)
hgb = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=None,
    max_iter=400,
    random_state=42
)

# choix du modele manuel
MODEL = logreg   # ou: hgb

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", MODEL),
])


# ======================
# 6) Entraînement
# ======================
pipe.fit(X_train, y_train)


# ======================
# 7) Évaluation
# ======================
y_pred = pipe.predict(X_test)

print("=== Confusion matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification report ===")
print(classification_report(y_test, y_pred, digits=4))

# AUC si le modèle expose predict_proba
if hasattr(pipe.named_steps["model"], "predict_proba"):
    y_proba = pipe.predict_proba(X_test)[:, 1]
    print("\n=== ROC AUC ===")
    print(f"{roc_auc_score(y_test, y_proba):.4f}")

# ======================
# 8) Sauvegarde du modèle
# ======================
OUT_PATH = "depression_classifier.joblib"
joblib.dump(pipe, OUT_PATH)
print(f"\nModèle sauvegardé: {OUT_PATH}")