import json
from shared.schema import TranscriptSegment


def _fmt_time_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _fmt_time_vtt(seconds: float) -> str:
    return _fmt_time_srt(seconds).replace(",", ".")


def export_json(segments: list[TranscriptSegment]) -> str:
    return json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=2)


def export_txt(segments: list[TranscriptSegment]) -> str:
    lines = []
    for s in segments:
        lines.append(f"[{s.participant_id}] ({s.start_time:.2f}s) {s.text}")
    return "\n".join(lines)


def export_srt(segments: list[TranscriptSegment]) -> str:
    blocks = []
    for i, s in enumerate(segments, 1):
        start = _fmt_time_srt(s.start_time)
        end = _fmt_time_srt(s.end_time)
        blocks.append(f"{i}\n{start} --> {end}\n[{s.participant_id}] {s.text}")
    return "\n\n".join(blocks) + "\n"


def export_vtt(segments: list[TranscriptSegment]) -> str:
    header = "WEBVTT\n\n"
    blocks = []
    for s in segments:
        start = _fmt_time_vtt(s.start_time)
        end = _fmt_time_vtt(s.end_time)
        blocks.append(f"{start} --> {end}\n[{s.participant_id}] {s.text}")
    return header + "\n\n".join(blocks) + "\n"
