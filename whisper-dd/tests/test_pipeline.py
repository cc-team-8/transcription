import numpy as np
import pytest
from unittest.mock import MagicMock
from client.pipeline import AudioPipeline
from shared.schema import TranscriptSegment


@pytest.fixture
def mock_pipeline():
    mock_vad = MagicMock()
    mock_asr = MagicMock()

    mock_vad.get_speech_segments.return_value = [(0, 16000)]

    dummy_seg = TranscriptSegment(
        session_id="s1",
        participant_id="p1",
        start_time=0.0,
        end_time=1.0,
        text="테스트",
        confidence=0.9,
    )
    mock_asr.transcribe.return_value = [dummy_seg]

    pipeline = AudioPipeline(
        vad=mock_vad,
        asr=mock_asr,
        session_id="s1",
        participant_id="p1",
        chunk_duration=1.0,
        sr=16000,
    )
    return pipeline, mock_vad, mock_asr


def test_process_chunk_calls_vad_and_asr(mock_pipeline):
    pipeline, mock_vad, mock_asr = mock_pipeline
    audio = np.zeros(16000, dtype=np.float32)
    segments = pipeline.process_chunk(audio, time_offset=0.0)
    mock_vad.get_speech_segments.assert_called_once()
    mock_asr.transcribe.assert_called_once()
    assert len(segments) == 1
    assert segments[0].text == "테스트"


def test_no_speech_skips_asr(mock_pipeline):
    pipeline, mock_vad, mock_asr = mock_pipeline
    mock_vad.get_speech_segments.return_value = []
    audio = np.zeros(16000, dtype=np.float32)
    segments = pipeline.process_chunk(audio, time_offset=0.0)
    mock_asr.transcribe.assert_not_called()
    assert segments == []
