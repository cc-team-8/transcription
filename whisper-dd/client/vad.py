import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps


class VADEngine:
    def __init__(self, threshold: float = 0.5):
        self.model = load_silero_vad()
        self.threshold = threshold

    def get_speech_segments(
        self, audio: np.ndarray, sr: int = 16000
    ) -> list[tuple[int, int]]:
        """Return list of (start_sample, end_sample) speech segments."""
        tensor = torch.from_numpy(audio)
        timestamps = get_speech_timestamps(
            tensor,
            self.model,
            sampling_rate=sr,
            threshold=self.threshold,
            return_seconds=False,
        )
        return [(t["start"], t["end"]) for t in timestamps]
