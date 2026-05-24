import numpy as np


def normalize_audio(audio: np.ndarray, src_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Convert arbitrary audio array to target_sr mono float32."""
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    if src_sr != target_sr:
        original_len = len(audio)
        target_len = int(original_len * target_sr / src_sr)
        indices = np.linspace(0, original_len - 1, target_len)
        audio = np.interp(indices, np.arange(original_len), audio).astype(np.float32)

    audio = np.clip(audio, -1.0, 1.0)
    return audio
