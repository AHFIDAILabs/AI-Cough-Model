import joblib
import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model

os.makedirs("static/reports", exist_ok=True)

model = joblib.load("model/tb_xgb_model_calibrated.joblib")
scaler = joblib.load("model/scaler.joblib")
pca = joblib.load("model/pca_audio.joblib")

import pandas as pd
template = pd.read_csv("model/prediction_template.csv")
feature_columns = template.columns.tolist()

CLINICAL_FEATURES = ["weight", "fever", "age", "tb_prior_Pul", "smoke_lweek"]
AUDIO_SCALE = 0.005


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model.eval()
