import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from shared.schema import TranscriptSegment
from client.asr import ASREngine


@pytest.fixture
def mock_asr():
    with patch("client.asr.WhisperModel") as MockModel:
        mock_instance = MagicMock()
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 2.5
        mock_segment.text = " 안녕하세요"
        mock_segment.avg_logprob = -0.2
        mock_instance.transcribe.return_value = (iter([mock_segment]), MagicMock())
        MockModel.return_value = mock_instance
        yield ASREngine(model_size="tiny")


def test_transcribe_returns_segment(mock_asr, sample_session_id, sample_participant_id):
    audio = np.zeros(16000 * 3, dtype=np.float32)
    result = mock_asr.transcribe(
        audio,
        session_id=sample_session_id,
        participant_id=sample_participant_id,
        time_offset=0.0,
    )
    assert len(result) == 1
    seg = result[0]
    assert isinstance(seg, TranscriptSegment)
    assert seg.text == "안녕하세요"
    assert seg.participant_id == sample_participant_id
    assert seg.session_id == sample_session_id


def test_time_offset_applied(mock_asr, sample_session_id, sample_participant_id):
    audio = np.zeros(16000, dtype=np.float32)
    result = mock_asr.transcribe(
        audio,
        session_id=sample_session_id,
        participant_id=sample_participant_id,
        time_offset=10.0,
    )
    assert result[0].start_time == pytest.approx(10.0)
    assert result[0].end_time == pytest.approx(12.5)


def test_confidence_between_0_and_1(mock_asr, sample_session_id, sample_participant_id):
    audio = np.zeros(16000, dtype=np.float32)
    result = mock_asr.transcribe(
        audio,
        session_id=sample_session_id,
        participant_id=sample_participant_id,
        time_offset=0.0,
    )
    assert 0.0 <= result[0].confidence <= 1.0
