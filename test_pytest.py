"""
Plan de test - depression_classifier
=====================================
Cas d'usage identifies :
  1. Detecter la depression chez des etudiants (usage principal)
  2. Signaler les profils atypiques via le score d'anomalie
  3. Etre robuste aux donnees incompletes (valeurs manquantes)
  4. Respecter la logique metier (stress eleve -> plus de depression predite)

Seuils acceptables definis :
  - F1-score global >= 0.70
  - Recall sur les depressifs >= 0.65  (manquer un cas est plus grave qu'un faux positif)
  - Taux d'anomalies sur donnees normales <= 10 %
  - Outliers extremes (>+5 sigma) detectes a >= 80 %
"""

import sys
import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score

PROJECT = Path(__file__).parent / "atelier-information-et-documentation-main"
sys.path.insert(0, str(PROJECT))
from predict import predict_pipeline, NUM_COLS

# ===========================================================
# Fixtures
# ===========================================================

@pytest.fixture(scope="module")
def model():
    return joblib.load(PROJECT / "depression_classifier.joblib")

@pytest.fixture(scope="module")
def imputer():
    return joblib.load(PROJECT / "imputer.joblib")

@pytest.fixture(scope="module")
def scaler():
    return joblib.load(PROJECT / "scaler.joblib")

@pytest.fixture(scope="module")
def iso_forest():
    return joblib.load(PROJECT / "isolation_forest.joblib")

@pytest.fixture(scope="module")
def dataset():
    df = pd.read_csv(PROJECT / "student_lifestyle_100k.csv")
    X = df.drop(columns=["Depression", "Student_ID"], errors="ignore")
    y = df["Depression"].astype(int)
    return X, y, df

@pytest.fixture(scope="module")
def top_corr_feature(dataset):
    """Retourne (nom, signe) de la feature numerique la plus correlee avec Depression."""
    X, y, df = dataset
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].assign(Depression=y).corr()["Depression"].drop("Depression")
    feat = corr.abs().idxmax()
    return feat, float(corr[feat])


# ===========================================================
# Tests unitaires - structure et sorties
# ===========================================================

@pytest.mark.unit
def test_prediction_length(model, dataset):
    X, y, _ = dataset
    assert len(model.predict(X.iloc[:100])) == 100

@pytest.mark.unit
def test_prediction_is_binary(model, dataset):
    X, _, _ = dataset
    preds = model.predict(X.iloc[:200])
    assert set(preds).issubset({0, 1}), f"Valeurs inattendues : {set(preds)}"

@pytest.mark.unit
def test_all_artefacts_load(model, imputer, scaler, iso_forest):
    assert all(obj is not None for obj in [model, imputer, scaler, iso_forest])

@pytest.mark.unit
def test_pipeline_output_columns(dataset):
    X, _, _ = dataset
    out = predict_pipeline(X.iloc[:10])
    assert {"prediction", "anomaly_score", "is_anomaly"}.issubset(out.columns)

@pytest.mark.unit
def test_pipeline_handles_missing_values(dataset):
    """Le pipeline doit fonctionner meme avec des NaN dans les colonnes numeriques."""
    X, _, _ = dataset
    X_missing = X.iloc[:100].copy()
    X_missing.loc[X_missing.index[:20], NUM_COLS[0]] = np.nan
    out = predict_pipeline(X_missing)
    assert len(out) == 100


# ===========================================================
# Tests de performance globale
# ===========================================================

@pytest.mark.perf
def test_f1_score_global(model, dataset):
    X, y, _ = dataset
    f1 = f1_score(y, model.predict(X))
    assert f1 >= 0.70, f"F1 insuffisant : {f1:.4f} (seuil 0.70)"

@pytest.mark.perf
def test_recall_depressed(model, dataset):
    """Manquer un etudiant depressif est plus grave qu'un faux positif."""
    X, y, _ = dataset
    recall = recall_score(y, model.predict(X))
    assert recall >= 0.65, f"Recall insuffisant : {recall:.4f} (seuil 0.65)"


# ===========================================================
# Slice tests - comportement par segment
# ===========================================================

@pytest.mark.slice
def test_recall_on_depressed_population(model, dataset):
    """Sur les etudiants reellement depressifs, le recall doit rester bon."""
    X, y, _ = dataset
    mask = y == 1
    recall = recall_score(y[mask], model.predict(X[mask]))
    assert recall >= 0.65, f"Recall sur depressifs : {recall:.4f} (seuil 0.65)"

@pytest.mark.slice
def test_high_stress_group_has_more_depression(model, dataset, top_corr_feature):
    """Les etudiants du quartile superieur sur la feature la plus correlee
    doivent avoir un taux de depression predit plus eleve que la moyenne."""
    X, y, _ = dataset
    feat, sign = top_corr_feature
    if feat not in X.select_dtypes(include=[np.number]).columns:
        pytest.skip(f"'{feat}' non numerique")

    q75 = X[feat].quantile(0.75)
    mask_high = X[feat] >= q75

    rate_all  = model.predict(X).mean()
    rate_high = model.predict(X[mask_high]).mean()

    if sign > 0:
        assert rate_high >= rate_all, (
            f"'{feat}' correllee positivement : quartile haut devrait predire plus de depression "
            f"({rate_high:.3f} vs moyenne {rate_all:.3f})"
        )
    else:
        assert rate_high <= rate_all, (
            f"'{feat}' correlee negativement : quartile haut devrait predire moins de depression "
            f"({rate_high:.3f} vs moyenne {rate_all:.3f})"
        )

@pytest.mark.slice
def test_anomaly_score_impact_on_predictions(dataset):
    """Les entrees aux scores d'anomalie les plus bas (plus suspectes)
    ne doivent pas avoir un recall effondre par rapport au reste."""
    X, y, _ = dataset
    out = predict_pipeline(X)
    threshold = out["anomaly_score"].quantile(0.10)
    mask_suspicious = out["anomaly_score"] <= threshold

    recall_suspicious = recall_score(y[mask_suspicious], out.loc[mask_suspicious, "prediction"])
    recall_normal     = recall_score(y[~mask_suspicious], out.loc[~mask_suspicious, "prediction"])

    assert recall_suspicious >= recall_normal * 0.80, (
        f"Recall trop degrade sur les entrees suspectes : "
        f"{recall_suspicious:.3f} vs {recall_normal:.3f} (seuil : 80% du recall normal)"
    )


# ===========================================================
# Tests de logique metier
# ===========================================================

@pytest.mark.business
def test_monotonicity_top_feature(model, dataset, top_corr_feature):
    """Forcer la feature la plus correllee a son 10e percentile vs son 90e percentile
    doit changer le taux de depression predit dans la bonne direction."""
    X, y, _ = dataset
    feat, sign = top_corr_feature
    if feat not in X.select_dtypes(include=[np.number]).columns:
        pytest.skip(f"'{feat}' non numerique")

    X_low  = X.copy(); X_low[feat]  = X[feat].quantile(0.10)
    X_high = X.copy(); X_high[feat] = X[feat].quantile(0.90)

    rate_low  = model.predict(X_low).mean()
    rate_high = model.predict(X_high).mean()

    if sign > 0:
        assert rate_high > rate_low, (
            f"Augmenter '{feat}' devrait augmenter le taux de depression : "
            f"{rate_low:.3f} -> {rate_high:.3f}"
        )
    else:
        assert rate_high < rate_low, (
            f"Augmenter '{feat}' devrait diminuer le taux de depression : "
            f"{rate_low:.3f} -> {rate_high:.3f}"
        )


# ===========================================================
# Tests du score d'anomalie
# ===========================================================

@pytest.mark.anomaly
def test_normal_data_anomaly_rate(dataset):
    """Sur des donnees normales, le taux d'anomalies ne doit pas depasser 10 %."""
    X, _, _ = dataset
    out = predict_pipeline(X)
    rate = out["is_anomaly"].mean()
    assert rate <= 0.10, f"Trop d'anomalies sur donnees normales : {rate:.2%} (seuil 10%)"

@pytest.mark.anomaly
def test_extreme_outliers_flagged(dataset):
    """Des valeurs a +5 ecarts-types doivent etre flaggees comme anomalies."""
    X, _, _ = dataset
    num_cols = X.select_dtypes(include=[np.number]).columns
    X_ext = X.iloc[:100].copy()
    for col in num_cols:
        X_ext[col] = X[col].mean() + 5 * X[col].std()

    out = predict_pipeline(X_ext)
    rate = out["is_anomaly"].mean()
    assert rate >= 0.80, f"Outliers extremes non detectes : {rate:.2%} (seuil 80%)"

@pytest.mark.anomaly
def test_anomaly_score_in_valid_range(dataset):
    """Le score de l'IsolationForest doit rester dans [-1, 1]."""
    X, _, _ = dataset
    scores = predict_pipeline(X.iloc[:2000])["anomaly_score"]
    assert scores.min() >= -1.0 and scores.max() <= 1.0, (
        f"Score hors plage : min={scores.min():.3f}, max={scores.max():.3f}"
    )
