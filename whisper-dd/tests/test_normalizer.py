import numpy as np
import pytest
from client.normalizer import normalize_audio


def test_already_16k_mono_passthrough():
    audio = np.ones(16000, dtype=np.float32) * 0.1
    result = normalize_audio(audio, src_sr=16000)
    assert result.dtype == np.float32
    assert result.ndim == 1
    assert len(result) == 16000


def test_resample_from_44100():
    audio = np.random.randn(44100).astype(np.float32) * 0.1
    result = normalize_audio(audio, src_sr=44100)
    assert result.ndim == 1
    expected_len = int(len(audio) * 16000 / 44100)
    assert abs(len(result) - expected_len) < expected_len * 0.02


def test_stereo_to_mono():
    stereo = np.random.randn(16000, 2).astype(np.float32) * 0.1
    result = normalize_audio(stereo, src_sr=16000)
    assert result.ndim == 1
    assert len(result) == 16000


def test_output_clipped_to_minus1_plus1():
    loud = np.ones(16000, dtype=np.float32) * 5.0
    result = normalize_audio(loud, src_sr=16000)
    assert result.max() <= 1.0
    assert result.min() >= -1.0
