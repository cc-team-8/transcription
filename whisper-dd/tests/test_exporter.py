import json
import pytest
from hub.exporter import export_json, export_txt, export_srt, export_vtt
from shared.schema import TranscriptSegment


@pytest.fixture
def segments():
    return [
        TranscriptSegment("s1", "p01", 1.0, 3.5, "회의 시작합니다", 0.95),
        TranscriptSegment("s1", "p02", 5.0, 8.2, "안녕하세요", 0.88),
    ]


def test_export_json_parseable(segments):
    output = export_json(segments)
    data = json.loads(output)
    assert len(data) == 2
    assert data[0]["participant_id"] == "p01"
    assert data[0]["text"] == "회의 시작합니다"


def test_export_txt_contains_text_and_participant(segments):
    output = export_txt(segments)
    assert "[p01]" in output
    assert "회의 시작합니다" in output
    assert "[p02]" in output
    assert "안녕하세요" in output


def test_export_srt_format(segments):
    output = export_srt(segments)
    lines = output.strip().split("\n")
    assert lines[0] == "1"
    assert "-->" in lines[1]
    assert "회의 시작합니다" in lines[2]
    assert lines[4] == "2"


def test_export_vtt_starts_with_webvtt(segments):
    output = export_vtt(segments)
    assert output.startswith("WEBVTT")
    assert "-->" in output
    assert "안녕하세요" in output
