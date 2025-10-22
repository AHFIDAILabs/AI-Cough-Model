import numpy as np
import librosa
import warnings
import os
from pydub import AudioSegment
import io

warnings.filterwarnings('ignore')


def check_audio_quality(audio_path, min_duration=0.5, max_duration=15.0):

    try:
        audio_data = None
        sr = 16000

        try:
            audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e1:
            print(f"[INFO] Trying format conversion: {e1}")
            try:
                AudioSegment.converter = r"C:\Users\hp\Desktop\ffmpeg\ffmpeg.exe"
                audio = AudioSegment.from_file(audio_path)
                audio = audio.set_frame_rate(16000).set_channels(1)
                wav_io = io.BytesIO()
                audio.export(wav_io, format="wav")
                wav_io.seek(0)
                audio_data, sr = librosa.load(wav_io, sr=16000, mono=True)
            except Exception as e2:
                print(f"[ERROR] Could not load audio: {e2}")
                return False, "Could not load audio file. Please try recording again.", {}

        duration = len(audio_data) / sr

        if duration < min_duration:
            return False, f"🎤 Recording too short ({duration:.1f}s). Please cough for at least {min_duration}s.", {}

        if duration > max_duration:
            print(f"[WARNING] Long recording ({duration:.1f}s) - using first {max_duration}s")
            audio_data = audio_data[:int(max_duration * sr)]
            duration = max_duration


        max_amplitude = np.max(np.abs(audio_data))

        # reject if EXTREMELY quiet (complete silence)
        if max_amplitude < 0.01:  # Very low threshold - only rejects silence
            return False, "🔇 No sound detected. Please ensure your microphone is working and record your cough.", {}

        std_deviation = np.std(audio_data)

        if std_deviation < 0.001:
            return False, "🔇 No audio variation detected. Please check your microphone and record again.", {}

        try:

            hop_length = 512
            n_fft = 2048
            stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            # Cough frequency range
            freq_mask = (freqs >= 200) & (freqs <= 4000)
            cough_freq_energy = np.mean(magnitude[freq_mask, :])

            # Calculate RMS energy
            rms = librosa.feature.rms(y=audio_data)[0]
            peak_energy = np.max(rms)
            avg_energy = np.mean(rms)
            energy_contrast = peak_energy / (avg_energy + 1e-6)

            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            avg_centroid = np.mean(spectral_centroids)

            # Zero-crossing rate
            zcr = librosa.zero_crossings(audio_data).sum()
            zcr_rate = zcr / len(audio_data)

        except Exception as e:
            print(f"[WARNING] Metric calculation failed: {e}")
            # Use default values if calculation fails
            cough_freq_energy = 0.05
            peak_energy = max_amplitude
            energy_contrast = 2.0
            avg_centroid = 1000.0
            zcr_rate = 0.05

        metrics = {
            "duration": float(duration),
            "max_amplitude": float(max_amplitude),
            "std_deviation": float(std_deviation),
            "cough_freq_energy": float(cough_freq_energy),
            "zcr_rate": float(zcr_rate),
            "peak_energy": float(peak_energy),
            "energy_contrast": float(energy_contrast),
            "spectral_centroid": float(avg_centroid),
            "sample_rate": int(sr)
        }

        if max_amplitude >= 0.15 and cough_freq_energy >= 0.02:
            message = " Excellent cough recording! Audio quality is very good."
        elif max_amplitude >= 0.08:
            message = " Good cough recording! Audio quality is acceptable."
        else:
            message = " Cough recorded. Audio may be quiet but is usable for analysis."

        return True, message, metrics

    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Audio quality check failed: {error_msg}")
        return False, f"Error processing audio. Please try recording again.", {}


def get_quality_score(metrics):

    if not metrics:
        return 50  # Default moderate score

    score = 0

    # Duration score (0.5-10 seconds all acceptable)
    duration = metrics.get('duration', 0)
    if 1.0 <= duration <= 5.0:
        score += 25
    elif 0.5 <= duration <= 10.0:
        score += 20
    else:
        score += 15

    max_amp = metrics.get('max_amplitude', 0)
    if max_amp >= 0.2:
        score += 25
    elif max_amp >= 0.08:
        score += 20
    elif max_amp >= 0.03:
        score += 15
    else:
        score += 10

    freq_energy = metrics.get('cough_freq_energy', 0)
    if freq_energy >= 0.05:
        score += 25
    elif freq_energy >= 0.02:
        score += 20
    elif freq_energy >= 0.01:
        score += 15
    else:
        score += 10

    energy_contrast = metrics.get('energy_contrast', 0)
    if energy_contrast >= 3.0:
        score += 15
    elif energy_contrast >= 1.5:
        score += 12
    else:
        score += 10


    centroid = metrics.get('spectral_centroid', 0)
    if 500 <= centroid <= 4000:  # Very wide range
        score += 10
    else:
        score += 5


    return max(min(score, 100), 40)