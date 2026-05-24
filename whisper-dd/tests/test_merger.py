import pytest
from hub.merger import merge_segments
from shared.schema import TranscriptSegment


def make_seg(participant_id, start, end, text):
    return TranscriptSegment(
        session_id="s1",
        participant_id=participant_id,
        start_time=start,
        end_time=end,
        text=text,
        confidence=0.9,
    )


def test_merge_two_participants_sorted_by_start():
    segs = [
        make_seg("p02", 5.0, 8.0, "안녕하세요"),
        make_seg("p01", 1.0, 3.0, "회의 시작합니다"),
        make_seg("p02", 10.0, 12.0, "네 알겠습니다"),
        make_seg("p01", 7.0, 9.0, "의견 있으신가요"),
    ]
    merged = merge_segments(segs)
    starts = [s.start_time for s in merged]
    assert starts == sorted(starts)


def test_merge_preserves_participant_id():
    segs = [
        make_seg("p01", 1.0, 2.0, "A"),
        make_seg("p02", 0.5, 1.5, "B"),
    ]
    merged = merge_segments(segs)
    assert merged[0].participant_id == "p02"
    assert merged[1].participant_id == "p01"


def test_merge_empty_list():
    assert merge_segments([]) == []


def test_merge_single_participant():
    segs = [make_seg("p01", i * 2.0, i * 2.0 + 1.5, f"발화{i}") for i in range(3)]
    merged = merge_segments(segs)
    assert len(merged) == 3
