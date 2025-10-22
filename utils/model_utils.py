import os
import torch
import joblib
import numpy as np
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from scipy.io import wavfile
import librosa
import warnings

warnings.filterwarnings('ignore')

os.makedirs("static/reports", exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)

MODEL_PATH = "model/tb_xgb_model_balanced.joblib"
SCALER_PATH = "model/scaler.joblib"
PCA_PATH = "model/pca_audio.joblib"
TEMPLATE_PATH = "model/prediction_template.csv"

if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, PCA_PATH, TEMPLATE_PATH]):
    raise FileNotFoundError("One or more model files are missing.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)
template = pd.read_csv(TEMPLATE_PATH)
feature_columns = template.columns.tolist()

CLINICAL_FEATURES = ["smoke_lweek", "weight", "age", "fever", "tb_prior_Pul"]
AUDIO_SCALE = 0.4

print(f"[INFO] Model loaded: {len(feature_columns)} features expected")
print(f"[INFO] PCA components: {pca.n_components_}")
print(f"[INFO] Audio scale: {AUDIO_SCALE}")

device = "cuda" if torch.cuda.is_available() else "cpu"
_processor = None
_wav2vec_model = None
_model_loaded = False


def _load_wav2vec2():
    """Load Wav2Vec2 model on first use (lazy loading to prevent startup crash)"""
    global _processor, _wav2vec_model, _model_loaded

    if not _model_loaded:
        print("[INFO] Loading Wav2Vec2 model for the first time...")
        try:
            _processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            _wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            _wav2vec_model.to(device)
            _wav2vec_model.eval()
            _model_loaded = True
            print(f"[INFO] Wav2Vec2 loaded successfully on {device}")
        except Exception as e:
            print(f"[ERROR] Failed to load Wav2Vec2: {e}")
            raise

    return _processor, _wav2vec_model

# AUDIO PREPROCESSING
def load_and_preprocess_audio(audio_path, target_sr=16000):
    """Load and preprocess audio file for Wav2Vec2."""
    try:

        if audio_path.endswith('.wav'):
            sample_rate, audio_data = wavfile.read(audio_path)

            # Convert to float32
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0

            # Stereo to mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Resample if needed
            if sample_rate != target_sr:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
                sample_rate = target_sr
        else:
            # Use librosa for other formats (.webm, .mp3, etc.)
            audio_data, sample_rate = librosa.load(audio_path, sr=target_sr, mono=True)

        # Normalize to [-1, 1]
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        return audio_data, sample_rate

    except Exception as e:
        print(f"[ERROR] Audio preprocessing failed: {e}")
        raise

def extract_audio_features(audio_path: str):
    """
    Extracts RAW Wav2Vec2 embeddings (768-dim) from audio file.
    Returns raw embeddings - PCA will be applied in preprocessing.py
    """
    if not os.path.exists(audio_path):
        print(f"[ERROR] File not found: {audio_path}")
        return None

    try:
        print(f"[INFO] Processing: {os.path.basename(audio_path)}")

        processor, wav2vec_model = _load_wav2vec2()

        audio_data, sample_rate = load_and_preprocess_audio(audio_path, target_sr=16000)

        print(f"[INFO] Audio loaded: duration={len(audio_data) / sample_rate:.2f}s, sr={sample_rate}Hz")

        min_length = sample_rate
        if len(audio_data) < min_length:
            print(f"[WARNING] Audio too short, padding to 1s")
            audio_data = np.pad(audio_data, (0, min_length - len(audio_data)), mode='constant')

        inputs = processor(
            audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = wav2vec_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

        print(f"[INFO] Extracted embeddings: shape={embeddings.shape}, mean={embeddings.mean():.6f}")

        if embeddings.shape[0] != 768:
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

        return embeddings

    except Exception as e:
        print(f"[ERROR] Audio feature extraction failed for {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_tb(input_vector, threshold=0.80):

    try:
        expected_features = len(feature_columns)
        if len(input_vector) != expected_features:
            raise ValueError(
                f"Input has {len(input_vector)} features, expected {expected_features}"
            )

        input_vector = np.array(input_vector).reshape(1, -1)

        proba = model.predict_proba(input_vector)[0]
        tb_probability = float(proba[1])

        prediction = int(tb_probability >= threshold)

        # Confidence is the probability of predicted class
        confidence = float(proba[prediction])

        print(f"[PREDICTION] Class probabilities: [No TB: {proba[0]:.4f}, TB: {proba[1]:.4f}]")
        print(f"[PREDICTION] Predicted: {prediction}, Prob: {tb_probability:.4f}, Conf: {confidence:.4f}")

        return prediction, tb_probability, confidence

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_feature_importance():
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return {
                feature_columns[i]: float(importance[i])
                for i in range(len(feature_columns))
            }
        else:
            print("[WARNING] Model has no feature_importances_")
            return {}
    except Exception as e:
        print(f"[ERROR] Failed to get feature importance: {e}")
        return {}


def get_model_info():
    """Returns model information."""
    return {
        "model_type": type(model).__name__,
        "n_features": len(feature_columns),
        "feature_names": feature_columns,
        "pca_components": pca.n_components_,
        "audio_scale": AUDIO_SCALE,
        "device": device
    }
