import io
import base64
import soundfile as sf
from pydub import AudioSegment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

# FFmpeg path
AudioSegment.converter = r"C:\Users\hp\Desktop\ffmpeg\ffmpeg.exe"
os.environ["PATH"] += os.pathsep + r"C:\Users\hp\Desktop\ffmpeg"

def load_waveform(uploaded_file, target_sr=16000):
    try:
        audio_bytes = uploaded_file.read()
        ext = uploaded_file.filename.split('.')[-1].lower()

        # Load with pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)
        audio = audio.set_frame_rate(target_sr).set_channels(1)

        # Export to BytesIO WAV
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Read waveform
        waveform_np, sr = sf.read(wav_io)
        waveform_np = waveform_np.astype('float32')

        # Ensure mono
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.mean(axis=1)

        # Normalize between -1 and 1
        max_val = np.max(np.abs(waveform_np))
        if max_val > 0:
            waveform_np /= max_val

        return waveform_np, sr

    except Exception as e:
        print(f"[ERROR] Failed to load waveform from {uploaded_file.filename}: {e}")
        # Return 1 second of silence at 16kHz
        return np.zeros(target_sr, dtype=np.float32), target_sr

def generate_spectrogram_base64(waveform, sr=16000):
    try:
        if waveform is None or len(waveform) == 0:
            waveform = np.zeros(sr, dtype=np.float32)  # 1 sec silence

        buf = io.BytesIO()
        plt.figure(figsize=(8, 4))
        plt.specgram(waveform, Fs=sr, NFFT=1024, noverlap=512, cmap='inferno')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"[ERROR] Failed to generate spectrogram: {e}")
        return None
