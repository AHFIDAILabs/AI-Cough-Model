import numpy as np
import pandas as pd
import joblib
import os

# loading model artifacts
template = pd.read_csv("model/prediction_template.csv")
feature_columns = template.columns.tolist()
scaler = joblib.load("model/scaler.joblib")
pca = joblib.load("model/pca_audio.joblib")

CLINICAL_FEATURES = ["weight", "fever", "age", "tb_prior_Pul", "smoke_lweek"]
AUDIO_SCALE = 0.005

def prepare_input_vector(clinical_dict, audio_vector=None):
    input_data = template.copy().iloc[0:1]
    input_data[:] = 0
    for key, val in clinical_dict.items():
        if key in input_data.columns:
            input_data.at[0, key] = int(val) if key in ['fever', 'tb_prior_Pul', 'smoke_lweek'] else float(val)
    scaled_input = scaler.transform(input_data[feature_columns])

    if audio_vector is not None:
        audio_vector = np.array(audio_vector).reshape(1, -1)
        audio_vector = audio_vector * AUDIO_SCALE
        scaled_input = np.hstack([scaled_input, audio_vector])
        # Pad or trim to match template length
        if scaled_input.shape[1] < len(feature_columns):
            padding = np.zeros((1, len(feature_columns) - scaled_input.shape[1]))
            scaled_input = np.hstack([scaled_input, padding])
        elif scaled_input.shape[1] > len(feature_columns):
            scaled_input = scaled_input[:, :len(feature_columns)]

    return scaled_input.flatten()

def prepare_clinical_only_vector(clinical_dict):
    return prepare_input_vector(clinical_dict, audio_vector=None)

def analyze_feature_dominance(input_vector, clinical_dict):
    feature_values = {feature_columns[i]: input_vector[i] for i in range(len(input_vector))}
    clinical_values = [abs(feature_values.get(f, 0)) for f in CLINICAL_FEATURES if f in feature_values]
    audio_values = [abs(v) for k, v in feature_values.items() if k.startswith("PCA_Audio_")]
    clinical_mag = np.mean(clinical_values) if clinical_values else 0
    audio_mag = np.mean(audio_values) if audio_values else 0
    ratio = audio_mag / clinical_mag if clinical_mag > 0 else 0
    audio_dominance = ratio > 0.5
    return {"clinical_mag": clinical_mag, "audio_mag": audio_mag, "audio_dominance": audio_dominance, "ratio": ratio}
