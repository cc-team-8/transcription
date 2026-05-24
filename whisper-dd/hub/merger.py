from shared.schema import TranscriptSegment


def merge_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """참가자별 세그먼트를 start_time 기준으로 정렬·병합한다."""
    return sorted(segments, key=lambda s: s.start_time)
