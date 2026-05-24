import os
os.environ["TORCH_HOME"] = os.path.expanduser("~/.cache/torch")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

print("1. torch import...")
import torch
print("2. torch OK")

print("3. silero_vad import...")
from silero_vad import load_silero_vad, get_speech_timestamps
print("4. silero_vad import OK")

print("5. load_silero_vad()...")
model = load_silero_vad()
print("6. model loaded OK")

print("7. faster_whisper import...")
from faster_whisper import WhisperModel
print("8. faster_whisper OK")

print("9. WhisperModel load...")
asr_model = WhisperModel("base", device="cpu", compute_type="int8")
print("10. WhisperModel OK")
