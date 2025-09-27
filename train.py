import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve
)
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

DATA_PATH = r"balanced_tb_data.csv"
TEST_PATH = r"balanced_tb_data_clean.csv"
MODEL_DIR = "model"

df = pd.read_csv(DATA_PATH)
df_test = pd.read_csv(TEST_PATH)

columns_to_drop = ["weight_loss", "reported_cough_dur", "night_sweats", "heart_rate"]
df.drop(columns=[c for c in columns_to_drop if c in df.columns], inplace=True)
df_test.drop(columns=[c for c in columns_to_drop if c in df_test.columns], inplace=True)

df_test.replace({"Yes": 1, "No": 0, "Not sure": 0, "Male": 1, "Female": 0}, inplace=True)
for col in ["participant", "sound_prediction_score"]:
    if col in df_test.columns:
        df_test.drop(col, axis=1, inplace=True)

print(f"Train shape: {df.shape}, Test shape: {df_test.shape}")
print("\nTarget distribution:\n", df["tb_status"].value_counts())

np.random.seed(42)
noise_frac = 0.02
noise_idx = df.sample(frac=noise_frac, random_state=42).index
df.loc[noise_idx, "tb_status"] = 1 - df.loc[noise_idx, "tb_status"]

for i in range(5):
    df[f"noise_{i}"] = np.random.randn(len(df))
    df_test[f"noise_{i}"] = np.random.randn(len(df_test))


#Features + Split
clinical_features = ["smoke_lweek", "weight", "age", "fever", "tb_prior_Pul"]
audio_features = [c for c in df.columns if c.startswith("feat_")]
noise_features = [c for c in df.columns if c.startswith("noise_")]

df["audio_provided"] = 1
df_test["audio_provided"] = 1

missing_audio_frac = 0.05
if missing_audio_frac > 0:
    aug_idx = df.sample(frac=missing_audio_frac, random_state=42).index
    df_aug = df.loc[aug_idx].copy().reset_index(drop=True)
    for col in audio_features:
        if col in df_aug.columns:
            df_aug[col] = 0.0
    df_aug["audio_provided"] = 0
    df = pd.concat([df, df_aug], ignore_index=True)
    print(f"Added {len(df_aug)} synthetic missing-audio examples for robustness.")

audio_features = [c for c in df.columns if c.startswith("feat_")]
noise_features = [c for c in df.columns if c.startswith("noise_")]

ordered_feature_list = clinical_features + ["audio_provided"] + audio_features + noise_features
ordered_feature_list = [c for c in ordered_feature_list if c in df.columns]

X = df[ordered_feature_list].copy()
y = df["tb_status"].copy()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#Reducing Audio Dominance -  PCA + Scaling
pca_components = 10
audio_scale = 0.4
pca = PCA(n_components=pca_components, random_state=42)

audio_cols_in_X = [c for c in X_train.columns if c.startswith("feat_")]
if len(audio_cols_in_X) == 0:
    raise RuntimeError("No audio feature columns (feat_*) found in training data. PCA requires audio features.")

audio_train_raw = X_train[audio_cols_in_X].values
audio_val_raw = X_val[audio_cols_in_X].values
audio_test_raw = df_test[audio_cols_in_X].values if set(audio_cols_in_X).issubset(set(df_test.columns)) else np.zeros((len(df_test), len(audio_cols_in_X)))

audio_train_reduced = pca.fit_transform(audio_train_raw) * audio_scale
audio_val_reduced = pca.transform(audio_val_raw) * audio_scale
audio_test_reduced = pca.transform(audio_test_raw) * audio_scale

def build_balanced_matrix(df_subset, audio_reduced, clinical_feats=clinical_features, noise_feats=noise_features):
    clinical_block = df_subset[[c for c in clinical_feats if c in df_subset.columns]].values
    audio_provided_block = df_subset[["audio_provided"]].values if "audio_provided" in df_subset.columns else np.ones((len(df_subset), 1))
    noise_block = df_subset[[c for c in noise_feats if c in df_subset.columns]].values if len(noise_feats) > 0 else np.empty((len(df_subset), 0))
    return np.hstack([clinical_block, audio_provided_block, audio_reduced, noise_block])

X_train_balanced = build_balanced_matrix(X_train, audio_train_reduced)
X_val_balanced = build_balanced_matrix(X_val, audio_val_reduced)

df_test_for_balanced = df_test.copy()
if "audio_provided" not in df_test_for_balanced.columns:
    df_test_for_balanced["audio_provided"] = 1
X_test_balanced = build_balanced_matrix(df_test_for_balanced, audio_test_reduced)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_val_scaled = scaler.transform(X_val_balanced)
X_test_scaled = scaler.transform(X_test_balanced)

print(f"Balanced Training shape: {X_train_scaled.shape}, Validation shape: {X_val_scaled.shape}, External shape: {X_test_scaled.shape}")

xgb_base = XGBClassifier(
    n_estimators=150, learning_rate=0.1, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42
)

param_dist = {
    "n_estimators": [100, 150, 200],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

rs = RandomizedSearchCV(
    estimator=xgb_base, param_distributions=param_dist,
    n_iter=5, scoring="f1", cv=3, n_jobs=1, random_state=42, verbose=1
)
rs.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)

best_xgb = rs.best_estimator_
print(f"\nBest parameters: {rs.best_params_}")

y_val_pred = best_xgb.predict(X_val_scaled)
y_val_proba = best_xgb.predict_proba(X_val_scaled)[:, 1]

print("\nValidation Report (XGBoost)\n")
print(classification_report(y_val, y_val_pred))
ConfusionMatrixDisplay(confusion_matrix(y_val, y_val_pred)).plot(cmap="Oranges")
plt.title("XGBoost - Validation Confusion Matrix")
plt.show()

#Feature Importances
balanced_feature_names = []
for c in clinical_features:
    if c in ordered_feature_list:
        balanced_feature_names.append(c)
balanced_feature_names.append("audio_provided")
balanced_feature_names += [f"PCA_Audio_{i}" for i in range(pca_components)]
balanced_feature_names += [c for c in noise_features if c in ordered_feature_list]

if len(balanced_feature_names) != X_train_scaled.shape[1]:
    if len(balanced_feature_names) < X_train_scaled.shape[1]:
        extra_needed = X_train_scaled.shape[1] - len(balanced_feature_names)
        balanced_feature_names += [f"extra_{i}" for i in range(extra_needed)]
    else:
        balanced_feature_names = balanced_feature_names[:X_train_scaled.shape[1]]

feature_importance = pd.Series(best_xgb.feature_importances_, index=balanced_feature_names)
feature_importance = feature_importance.sort_values(ascending=False)

print("\nTop 10 Features by XGBoost Importance (Balanced):")
print(feature_importance.head(10))

plt.figure(figsize=(8,6))
sns.barplot(x=feature_importance.head(10).values, y=feature_importance.head(10).index, palette="magma")
plt.title("Top 10 Features (XGBoost, Balanced)")
plt.show()

#Compute Youden's J Threshold + pick deployment threshold
fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
youden_threshold = float(thresholds[best_idx])
deployment_threshold = 0.85

print(f"\nOptimal Youden's J Threshold: {youden_threshold:.4f}")
print(f"Recommended deployment threshold (saved): {deployment_threshold:.4f}")

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
with open(f"{MODEL_DIR}/threshold_youden.txt", "w") as f:
    f.write(str(youden_threshold))
with open(f"{MODEL_DIR}/threshold_deployment.txt", "w") as f:
    f.write(str(deployment_threshold))
with open(f"{MODEL_DIR}/thresholds.json", "w") as f:
    json.dump({"youden": youden_threshold, "deployment": deployment_threshold}, f)

joblib.dump(best_xgb, f"{MODEL_DIR}/tb_xgb_model_balanced.joblib")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
joblib.dump(pca, f"{MODEL_DIR}/pca_audio.joblib")

feature_importance_df = feature_importance.reset_index()
feature_importance_df.columns = ["Feature", "Importance"]
feature_importance_df.to_csv(f"{MODEL_DIR}/feature_importance.csv", index=False)

df_test["tb_proba"] = best_xgb.predict_proba(X_test_scaled)[:, 1]
df_test["tb_pred_youden"] = (df_test["tb_proba"] >= youden_threshold).astype(int)
df_test["tb_pred_deploy"] = (df_test["tb_proba"] >= deployment_threshold).astype(int)
df_test.to_csv(f"{MODEL_DIR}/solicited_test_predictions.csv", index=False)

template_df = pd.DataFrame([{f: 0 for f in balanced_feature_names}])
template_df.to_csv(f"{MODEL_DIR}/prediction_template.csv", index=False)

#metrics and validation summary

print(f"Final Validation F1: {f1_score(y_val, y_val_pred):.4f}, "
      f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}, "
      f"AUC: {roc_auc_score(y_val, y_val_proba):.4f}")

#External test evaluation
thresholds_to_check = sorted(list({float(youden_threshold), float(deployment_threshold), 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50}), reverse=True)

if "tb_status" in df_test.columns:
    rows = []
    for t in thresholds_to_check:
        y_test_pred = (df_test["tb_proba"] >= t).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(df_test["tb_status"], y_test_pred).ravel()
        except Exception:
            tp = int(((df_test["tb_status"] == 1) & (y_test_pred == 1)).sum())
            fn = int(((df_test["tb_status"] == 1) & (y_test_pred == 0)).sum())
            fp = int(((df_test["tb_status"] == 0) & (y_test_pred == 1)).sum())
            tn = int(((df_test["tb_status"] == 0) & (y_test_pred == 0)).sum())

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        rows.append({
            "threshold": t,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "sensitivity": sens, "specificity": spec,
            "precision": prec, "f1": f1, "accuracy": acc
        })

    df_metrics = pd.DataFrame(rows).sort_values("threshold", ascending=False)
    df_metrics.to_csv(f"{MODEL_DIR}/external_test_thresholds_metrics.csv", index=False)
    print("\nExternal test metrics saved to:", f"{MODEL_DIR}/external_test_thresholds_metrics.csv")
    print(df_metrics.to_string(index=False))

    y_test_pred_youden = (df_test["tb_proba"] >= youden_threshold).astype(int)
    print("\nExternal Test Classification Report (Youden's threshold):\n")
    print(classification_report(df_test["tb_status"], y_test_pred_youden))

    cm = confusion_matrix(df_test["tb_status"], y_test_pred_youden)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title("External Test - Confusion Matrix (Youden's threshold)")
    plt.show()

else:
    rows = []
    N = len(df_test)
    for t in thresholds_to_check:
        pred_pos = int((df_test["tb_proba"] >= t).sum())
        pred_neg = int((df_test["tb_proba"] < t).sum())
        rows.append({"threshold": t, "predicted_positive": pred_pos, "predicted_negative": pred_neg,
                     "pred_pos_rate": pred_pos / N, "pred_neg_rate": pred_neg / N})

    df_counts = pd.DataFrame(rows).sort_values("threshold", ascending=False)
    df_counts.to_csv(f"{MODEL_DIR}/external_test_thresholds_metrics_no_labels.csv", index=False)
    print("\nExternal test has NO 'tb_status' labels. Counts by threshold saved to:", f"{MODEL_DIR}/external_test_thresholds_metrics_no_labels.csv")
    print(df_counts.to_string(index=False))

    plt.figure(figsize=(8,4))
    plt.hist(df_test["tb_proba"], bins=50)
    plt.title("External Test - Predicted Probability Distribution (tb_proba)")
    plt.xlabel("tb_proba")
    plt.ylabel("count")
    plt.show()


