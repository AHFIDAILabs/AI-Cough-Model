import io
import base64
import torch
import soundfile as sf
from pydub import AudioSegment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# FFmpeg path (update path as needed)
AudioSegment.converter = r"C:\Users\hp\Desktop\ffmpeg\ffmpeg.exe"
os.environ["PATH"] += os.pathsep + r"C:\Users\hp\Desktop\ffmpeg"

def load_waveform(uploaded_file):
    audio_bytes = uploaded_file.read()
    ext = uploaded_file.filename.split('.')[-1].lower()
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)
    wav_io = io.BytesIO()
    audio.set_frame_rate(16000).set_channels(1).export(wav_io, format="wav")
    wav_io.seek(0)
    waveform_np, sr = sf.read(wav_io)
    waveform = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)
    return waveform, sr

def generate_spectrogram_base64(waveform):
    buf = io.BytesIO()
    plt.figure(figsize=(8, 4))
    plt.specgram(waveform.squeeze().numpy(), Fs=16000, NFFT=1024, noverlap=512, cmap='inferno')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
