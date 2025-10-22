from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import datetime
import uuid
import traceback
import os
import json

from utils.audio_utils import load_waveform, generate_spectrogram_base64
from utils.preprocessing import (
    prepare_input_vector,
    prepare_clinical_only_vector,
    validate_input,
    analyze_feature_dominance
)
from utils.report_utils import generate_pdf_report
from utils.model_utils import (
    extract_audio_features,
    predict_tb,
    get_model_info,
    AUDIO_SCALE
)


app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Load deployment threshold
try:
    with open("model/thresholds.json", "r") as f:
        thresholds = json.load(f)
        deployment_thresh = thresholds.get("deployment", 0.80)
except:
    deployment_thresh = 0.80

print(f"\n{'=' * 60}")
print("TB PREDICTION SYSTEM INITIALIZED")
print(f"{'=' * 60}")
print(f"Deployment threshold: {deployment_thresh}")
print(f"Audio scale factor: {AUDIO_SCALE}")

model_info = get_model_info()
print(f"Model type: {model_info['model_type']}")
print(f"Expected features: {model_info['n_features']}")
print(f"PCA components: {model_info['pca_components']}")
print(f"{'=' * 60}\n")


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
        print("\n" + "=" * 60)
        print("NEW PREDICTION REQUEST")
        print("=" * 60)

        #EXTRACT & VALIDATE CLINICAL DATA
        try:
            clinical_dict = {
                "smoke_lweek": int(request.form['smoke_lweek']),
                "weight": float(request.form['weight']),
                "age": int(request.form['age']),
                "fever": int(request.form['fever']),
                "tb_prior_Pul": int(request.form['tb_prior_Pul']),
            }
        except (KeyError, ValueError) as e:
            return jsonify({"error": f"Invalid clinical data: {str(e)}"}), 400

        print(f"[INFO] Clinical Data:")
        for key, val in clinical_dict.items():
            print(f"  {key}: {val}")

        # Validate
        is_valid, error_msg = validate_input(clinical_dict)
        if not is_valid:
            return jsonify({"error": f"Validation error: {error_msg}"}), 400


        from utils.audio_quality_checker import check_audio_quality, get_quality_score

        audio_files = request.files.getlist('audio_file')
        raw_audio_features = None
        spectrogram_base64 = None
        audio_quality_info = {}

        print(f"\n[INFO] Received {len(audio_files)} audio file(s)")

        os.makedirs("temp_audio", exist_ok=True)

        if len(audio_files) > 0 and audio_files[0].filename:
            audio_file = audio_files[0]  # Take only the first cough

            try:
                temp_filename = f"temp_{uuid.uuid4().hex[:8]}_{audio_file.filename}"
                temp_path = os.path.join("temp_audio", temp_filename)
                audio_file.save(temp_path)

                print(f"[INFO] Processing audio: {audio_file.filename}")

                # quality checking
                is_valid, quality_msg, metrics = check_audio_quality(temp_path)
                quality_score = get_quality_score(metrics)

                print(f"[QUALITY CHECK] Valid: {is_valid}, Score: {quality_score}/100")
                print(f"[QUALITY CHECK] Message: {quality_msg}")

                if not is_valid:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    return jsonify({
                        "error": "audio_quality_failed",
                        "message": quality_msg,
                        "quality_score": quality_score,
                        "metrics": metrics,
                        "retry": True
                    }), 400

                raw_audio_features = extract_audio_features(temp_path)

                if raw_audio_features is not None:
                    print(f"[SUCCESS] Extracted features from audio")
                    audio_quality_info = {
                        "score": quality_score,
                        "metrics": metrics,
                        "message": quality_msg
                    }
                else:
                    print(f"[WARNING] Failed to extract features")

                # Generate spectrogram
                try:
                    audio_file.seek(0)
                    waveform, sr = load_waveform(audio_file)
                    spectrogram_base64 = generate_spectrogram_base64(waveform, sr)
                    print("[INFO] Spectrogram generated")
                except Exception as e:
                    print(f"[WARNING] Spectrogram failed: {e}")

                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                print(f"[ERROR] Processing audio failed: {e}")
                traceback.print_exc()
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            print("\n[WARNING] No audio provided - using clinical-only mode")


        print("\n[INFO] Running predictions...")

        print("[INFO] Preparing clinical-only vector...")
        input_vector_clinical = prepare_clinical_only_vector(clinical_dict)
        pred_clinical, proba_clinical, conf_clinical = predict_tb(input_vector_clinical, deployment_thresh)

        # Clinical + Audio prediction
        if raw_audio_features is not None:
            print("[INFO] Preparing full (clinical + audio) vector...")
            input_vector_full = prepare_input_vector(clinical_dict, audio_vector=raw_audio_features)
            pred_full, proba_full, conf_full = predict_tb(input_vector_full, deployment_thresh)

            dominance = analyze_feature_dominance(input_vector_full, clinical_dict)
            audio_influence = proba_full - proba_clinical

            print(f"[INFO] Clinical-only probability: {proba_clinical:.4f}")
            print(f"[INFO] Full (with audio) probability: {proba_full:.4f}")
            print(f"[INFO] Audio influence: {audio_influence:.4f}")

            used_audio = True

            print(f"[INFO] ✓ Using audio features in final prediction")
            final_prediction = pred_full
            final_proba = proba_full
            final_confidence = conf_full

        else:
            used_audio = False
            final_prediction = pred_clinical
            final_proba = proba_clinical
            final_confidence = conf_clinical
            dominance = {"clinical_mag": 0, "audio_mag": 0, "ratio": 0, "audio_dominance": False}
            audio_influence = 0.0

            print(f"[INFO] No audio provided - using clinical-only prediction")

        print(f"\n[FINAL RESULT]")
        print(f"  Prediction: {'Probable TB' if final_prediction else 'Unlikely TB'}")
        print(f"  TB Probability: {final_proba:.4f}")
        print(f"  Confidence: {final_confidence:.4f}")
        print(f"  Audio used: {used_audio}")

        # GENERATE RECOMMENDATION
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        clinical_summary = (
            f"Weight: {clinical_dict['weight']} kg, "
            f"Age: {clinical_dict['age']} years, "
            f"Fever: {'Yes' if clinical_dict['fever'] else 'No'}, "
            f"Prior TB: {'Yes' if clinical_dict['tb_prior_Pul'] else 'No'}, "
            f"Smoking: {'Yes' if clinical_dict['smoke_lweek'] else 'No'}"
        )

        if final_prediction == 1:
            recommendation = f"""
Based on your clinical profile ({clinical_summary}){' and cough audio analysis' if used_audio else ''}, 
our AI screening tool suggests a PROBABLE likelihood of active tuberculosis.

IMPORTANT NEXT STEPS:
1. Visit a recognized health facility or TB treatment center immediately
2. Request confirmatory tests: chest X-ray, sputum smear microscopy, or GeneXpert MTB/RIF
3. Avoid close contact with vulnerable individuals until diagnosis is confirmed
4. Follow all medical advice provided by your healthcare provider

Remember: This is a screening tool, not a diagnostic test. Only laboratory tests can confirm TB.
"""
        else:
            recommendation = f"""
Based on your clinical profile ({clinical_summary}){' and cough audio analysis' if used_audio else ''} combined with the cough sound, 
our AI screening tool suggests tuberculosis is UNLIKELY at this time.

However, please note:
- Monitor your symptoms closely
- If symptoms persist or worsen (especially cough over 2 weeks, fever, weight loss), consult a healthcare provider
- This tool provides risk assessment, not definitive diagnosis
- When in doubt, always seek professional medical evaluation

Stay healthy and maintain good respiratory hygiene practices.
"""

        disclaimer = (
            "MEDICAL DISCLAIMER: This is an AI-based screening tool designed to support healthcare "
            "workers in prioritizing patients for further testing. It is NOT a substitute for professional "
            "medical diagnosis, clinical judgment, or laboratory confirmation. Always consult qualified "
            "healthcare providers for medical decisions."
        )

        result = {
            "date": now,
            "prediction": "Probable TB" if final_prediction else "Unlikely TB",
            "confidence": f"{final_confidence * 100:.2f}%",
            "probability": f"{final_proba * 100:.2f}%",
            "recommendation": recommendation,
            "disclaimer": disclaimer,
            "clinical_summary": clinical_summary,
            "audio_used": used_audio,
            "audio_quality": audio_quality_info if used_audio else None,  # added as requested
            "debug_info": {
                "clinical_only_prob": f"{proba_clinical:.4f}",
                "full_prob": f"{proba_full:.4f}" if raw_audio_features is not None else "N/A",
                "audio_influence": f"{audio_influence:.4f}" if raw_audio_features is not None else "0.0000",
                "used_audio": str(used_audio),
                "audio_dominance": str(dominance.get('audio_dominance', False)),
                "threshold": f"{deployment_thresh:.2f}"
            }
        }

        if spectrogram_base64:
            result["spectrogram_base64"] = spectrogram_base64

        # GENERATE PDF REPORT

        try:
            unique_id = str(uuid.uuid4())[:8]
            pdf_path = generate_pdf_report(result, unique_id)
            if pdf_path and os.path.exists(pdf_path):
                result["report_link"] = f"/static/reports/{unique_id}.pdf"
                print(f"[INFO] PDF report generated: {unique_id}.pdf")
        except Exception as e:
            print(f"[WARNING] PDF generation failed: {e}")

        print("=" * 60)
        print("PREDICTION COMPLETED SUCCESSFULLY")
        print("=" * 60 + "\n")

        return jsonify(result)

    except Exception as e:
        print("\n" + "=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60 + "\n")
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "details": "Check server logs for details"
        }), 500


@app.route('/static/reports/<filename>')
def download_report(filename):
    return send_from_directory("static/reports", filename, as_attachment=True)


@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.datetime.now().isoformat()
    })


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
