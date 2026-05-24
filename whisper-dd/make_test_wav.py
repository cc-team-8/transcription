import numpy as np
import soundfile as sf
import os

os.makedirs('recordings', exist_ok=True)
sr = 16000
t = np.linspace(0, 2, sr * 2)
audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
sf.write('recordings/test.wav', audio, sr)
print('생성 완료: recordings/test.wav')
