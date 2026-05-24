import numpy as np
import pytest
from client.vad import VADEngine


@pytest.fixture(scope="module")
def vad():
    return VADEngine()


def test_returns_sample_indices(vad, sample_audio_16k):
    segments = vad.get_speech_segments(sample_audio_16k, sr=16000)
    for start, end in segments:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert 0 <= start < end <= len(sample_audio_16k)


def test_silence_only_returns_no_segments(vad):
    silence = np.zeros(16000, dtype=np.float32)
    segments = vad.get_speech_segments(silence, sr=16000)
    assert len(segments) == 0


def test_output_is_list_of_tuples(vad, sample_audio_16k):
    segments = vad.get_speech_segments(sample_audio_16k, sr=16000)
    assert isinstance(segments, list)
    for item in segments:
        assert isinstance(item, tuple)
        assert len(item) == 2
