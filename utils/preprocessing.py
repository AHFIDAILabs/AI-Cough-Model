import numpy as np
import pandas as pd
import joblib
import os

TEMPLATE_PATH = "model/prediction_template.csv"
SCALER_PATH = "model/scaler.joblib"
PCA_PATH = "model/pca_audio.joblib"

for path in [TEMPLATE_PATH, SCALER_PATH, PCA_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

template = pd.read_csv(TEMPLATE_PATH)
feature_columns = template.columns.tolist()
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)

print(f"[INFO] Preprocessing ready: {len(feature_columns)} features")

CLINICAL_FEATURES = ["smoke_lweek", "weight", "age", "fever", "tb_prior_Pul"]
AUDIO_SCALE = 0.4
DEBUG_MODE = True



def prepare_input_vector(clinical_dict, audio_vector=None):

    try:
        if DEBUG_MODE:
            print("\n[DEBUG] ========== Building Input Vector ==========")
            print(f"[DEBUG] Clinical data: {clinical_dict}")
            print(f"[DEBUG] Audio provided: {audio_vector is not None}")

        # Initialize with template
        input_row = template.copy().iloc[0:1]
        input_row[:] = 0

        # Fill clinical features IN EXACT ORDER
        for feat in CLINICAL_FEATURES:
            if feat in clinical_dict:
                val = clinical_dict[feat]
                if feat in input_row.columns:
                    input_row.at[0, feat] = float(val)
                    if DEBUG_MODE:
                        print(f"[DEBUG] {feat} = {val}")

        # Set audio_provided flag
        audio_provided = 1 if audio_vector is not None else 0
        if 'audio_provided' in input_row.columns:
            input_row.at[0, 'audio_provided'] = audio_provided
            if DEBUG_MODE:
                print(f"[DEBUG] audio_provided = {audio_provided}")

        if audio_vector is not None and len(audio_vector) > 0:
            try:
                audio_vector = np.array(audio_vector).reshape(1, -1)

                if DEBUG_MODE:
                    print(f"[DEBUG] Raw audio shape: {audio_vector.shape}")
                    print(
                        f"[DEBUG] Audio stats: min={audio_vector.min():.4f}, max={audio_vector.max():.4f}, mean={audio_vector.mean():.4f}")

                # Apply PCA (768 -> 10 dimensions)
                audio_pca = pca.transform(audio_vector)

                # Apply scaling AFTER PCA (matching training)
                audio_pca_scaled = audio_pca * AUDIO_SCALE

                if DEBUG_MODE:
                    print(f"[DEBUG] PCA output shape: {audio_pca.shape}")
                    print(
                        f"[DEBUG] Scaled PCA stats: min={audio_pca_scaled.min():.4f}, max={audio_pca_scaled.max():.4f}, mean={audio_pca_scaled.mean():.4f}")

                # Insert into template
                for i in range(audio_pca_scaled.shape[1]):
                    col_name = f"PCA_Audio_{i}"
                    if col_name in input_row.columns:
                        input_row.at[0, col_name] = audio_pca_scaled[0, i]

            except Exception as e:
                print(f"[ERROR] Audio processing failed: {e}")
                import traceback
                traceback.print_exc()

        # Convert to array
        input_array = input_row[feature_columns].values

        if DEBUG_MODE:
            print(f"[DEBUG] Pre-scaling shape: {input_array.shape}")
            print(f"[DEBUG] Non-zero features: {np.count_nonzero(input_array)}")

        # Apply StandardScaler
        scaled_input = scaler.transform(input_array)

        if DEBUG_MODE:
            print(f"[DEBUG] Scaled shape: {scaled_input.shape}")
            print(
                f"[DEBUG] Scaled stats: min={scaled_input.min():.4f}, max={scaled_input.max():.4f}, mean={scaled_input.mean():.4f}")
            print("[DEBUG] ========== Vector Complete ==========\n")

        # Verify final shape
        if scaled_input.shape[1] != len(feature_columns):
            raise ValueError(f"Shape mismatch! Got {scaled_input.shape[1]}, expected {len(feature_columns)}")

        return scaled_input.flatten()

    except Exception as e:
        print(f"[ERROR] prepare_input_vector failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def prepare_clinical_only_vector(clinical_dict):
    return prepare_input_vector(clinical_dict, audio_vector=None)


def validate_input(clinical_dict):
    try:
        age = int(clinical_dict.get("age", 0))
        if not (1 <= age <= 120):
            return False, "Age must be 1-120"

        weight = float(clinical_dict.get("weight", 0))
        if not (1 <= weight <= 300):
            return False, "Weight must be 1-300 kg"

        for key in ["fever", "tb_prior_Pul", "smoke_lweek"]:
            val = int(clinical_dict.get(key, 0))
            if val not in [0, 1]:
                return False, f"{key} must be 0 or 1"

        return True, "Valid"

    except Exception as e:
        return False, str(e)


def analyze_feature_dominance(input_vector, clinical_dict):
    try:
        feature_values = {feature_columns[i]: input_vector[i] for i in range(len(input_vector))}

        clinical_vals = [abs(feature_values.get(f, 0)) for f in CLINICAL_FEATURES if f in feature_values]
        audio_vals = [abs(feature_values.get(f"PCA_Audio_{i}", 0)) for i in range(pca.n_components_)]

        clinical_mag = np.mean(clinical_vals) if clinical_vals else 0.0
        audio_mag = np.mean(audio_vals) if audio_vals else 0.0
        ratio = audio_mag / clinical_mag if clinical_mag > 0 else 0.0
        audio_dominance = ratio > 1.5

        return {
            "clinical_mag": float(clinical_mag),
            "audio_mag": float(audio_mag),
            "ratio": float(ratio),
            "audio_dominance": bool(audio_dominance)
        }
    except:
        return {"clinical_mag": 0, "audio_mag": 0, "ratio": 0, "audio_dominance": False}
