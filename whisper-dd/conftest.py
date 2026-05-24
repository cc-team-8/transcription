import os
import numpy as np
import pytest

# WSL2: prevent SIGBUS from mmap on NTFS by keeping torch cache on Linux fs
os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))


@pytest.fixture
def sample_audio_16k():
    """16kHz mono float32: 1s silence + 0.5s 440Hz tone + 0.5s silence"""
    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(sr // 2) / sr).astype(np.float32)
    silence2 = np.zeros(sr // 2, dtype=np.float32)
    return np.concatenate([silence, tone, silence2])


@pytest.fixture
def sample_session_id():
    return "meet-20260523-test"


@pytest.fixture
def sample_participant_id():
    return "p01"
