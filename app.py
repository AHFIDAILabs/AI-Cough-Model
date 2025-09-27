from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import datetime
import uuid
import torch
import joblib
import traceback
from utils.audio_utils import load_waveform, generate_spectrogram_base64
from utils.preprocessing import prepare_input_vector, prepare_clinical_only_vector, analyze_feature_dominance
from utils.report_utils import generate_pdf_report
from utils.model_utils import model, scaler, pca, template, processor, wav2vec_model, CLINICAL_FEATURES, AUDIO_SCALE

calibrated_model = joblib.load("model/tb_xgb_model_balanced.joblib")
scaler = joblib.load("model/scaler.joblib")

deployment_thresh = 0.80
print(f"[INFO] Using deployment threshold: {deployment_thresh}")

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/faq')
def faq():
    return render_template("faq.html")

@app.route('/predict', methods=["GET"])
def predict_form():
    return render_template("prediction.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        weight = float(request.form['weight'])
        age = int(request.form['age'])
        fever = int(request.form['fever'])
        tb_prior_Pul = int(request.form['tb_prior_Pul'])
        smoke_lweek = int(request.form['smoke_lweek'])

        if age < 1 or age > 100:
            return jsonify({"error": "Invalid input: Age must be between 1 and 100."})
        if weight <= 0:
            return jsonify({"error": "Invalid input: Weight must be greater than 0."})
        if fever not in [0, 1] or tb_prior_Pul not in [0, 1] or smoke_lweek not in [0, 1]:
            return jsonify({"error": "Invalid input: Yes/No fields must be 0 (No) or 1 (Yes)."})

        audio_file = request.files.get('audio_file')
        embeddings = None
        spectrogram_base64 = None

        if audio_file and audio_file.filename:
            try:
                waveform, sr = load_waveform(audio_file)
                inputs = processor(
                    waveform.squeeze().numpy(),
                    sampling_rate=sr,
                    return_tensors="pt",
                    padding=True
                )
                with torch.no_grad():
                    outputs = wav2vec_model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                spectrogram_base64 = generate_spectrogram_base64(waveform)
                print("Audio processed successfully")
            except Exception as e:
                print(f"Audio processing error: {e}")
                embeddings = None
                spectrogram_base64 = None


        clinical_dict = {
            "weight": weight,
            "fever": fever,
            "age": age,
            "tb_prior_Pul": tb_prior_Pul,
            "smoke_lweek": smoke_lweek
        }

        input_vector_clinical = prepare_clinical_only_vector(clinical_dict)
        proba_clinical_only = calibrated_model.predict_proba([input_vector_clinical])[0][1]
        prediction_clinical = 1 if proba_clinical_only >= deployment_thresh else 0

        # Clinical + Audio prediction
        input_vector_full = prepare_input_vector(clinical_dict, audio_vector=embeddings)
        proba_full = calibrated_model.predict_proba([input_vector_full])[0][1]
        prediction_full = 1 if proba_full >= deployment_thresh else 0

        # Feature dominance analysis
        dominance_analysis = analyze_feature_dominance(input_vector_full, clinical_dict)

        audio_influence = proba_full - proba_clinical_only
        used_audio = prediction_clinical == prediction_full and not dominance_analysis['audio_dominance'] and abs(audio_influence) <= 0.10

        final_proba = proba_full if used_audio else proba_clinical_only
        prediction = prediction_full if used_audio else prediction_clinical

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        #Recommendations
        if prediction == 1:
            recommendation = f"""
Based on your details — weight {weight} kg, fever ({'Yes' if fever else 'No'}), 
age {age}, prior pulmonary TB history ({'Yes' if tb_prior_Pul else 'No'}), 
smoking habit ({'Yes' if smoke_lweek else 'No'}) — {'combined with your cough sound' if used_audio and embeddings is not None else ''},
our system suggests a probable likelihood of active tuberculosis.

It is strongly advised that you visit a recognized health facility or TB treatment center.
Further tests such as chest X-ray, sputum smear, or GeneXpert may be required to confirm the diagnosis.
"""
        else:
            recommendation = f"""
Based on your details — weight {weight} kg, fever ({'Yes' if fever else 'No'}), 
age {age}, prior pulmonary TB history ({'Yes' if tb_prior_Pul else 'No'}), 
smoking habit ({'Yes' if smoke_lweek else 'No'}) — {'combined with your cough sound' if used_audio and embeddings is not None else ''},
the findings do not strongly suggest active tuberculosis at this time.

Monitor symptoms and seek medical advice if needed.
"""


        result = {
            "date": now,
            "prediction": "Probable TB" if prediction else "Unlikely TB",
            "confidence": f"{final_proba * 100:.2f}%",
            "recommendation": recommendation,
            "disclaimer": "This is an AI-based estimate. Always consult a healthcare provider.",
            "debug_info": {
                "clinical_only_prob": f"{proba_clinical_only:.4f}",
                "full_prob": f"{proba_full:.4f}",
                "audio_influence": f"{audio_influence:.4f}",
                "used_audio": str(used_audio),
                "audio_dominance": str(dominance_analysis['audio_dominance']),
                "audio_clinical_ratio": f"{dominance_analysis['ratio']:.6f}"
            }
        }

        if spectrogram_base64:
            result["spectrogram_base64"] = spectrogram_base64

        unique_id = str(uuid.uuid4())[:8]
        pdf_path = generate_pdf_report(result, unique_id)
        if pdf_path:
            result["report_link"] = f"/static/reports/{unique_id}.pdf"

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction Error: {str(e)}"})


@app.route('/static/reports/<filename>')
def download_report(filename):
    return send_from_directory("static/reports", filename)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
