from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from shared.schema import TranscriptSegment
from hub.merger import merge_segments
from hub.exporter import export_json, export_txt, export_srt, export_vtt

app = FastAPI(title="Meeting Transcription Hub")

# 인메모리 저장소: session_id → [TranscriptSegment]
_store: dict[str, list[TranscriptSegment]] = defaultdict(list)


class SegmentIn(BaseModel):
    session_id: str
    participant_id: str
    start_time: float
    end_time: float
    text: str
    confidence: float


@app.post("/segments", status_code=201)
def post_segment(body: SegmentIn):
    seg = TranscriptSegment(**body.model_dump())
    _store[body.session_id].append(seg)
    return {"status": "ok"}


@app.get("/sessions/{session_id}/transcript")
def get_transcript(session_id: str):
    segs = _store.get(session_id, [])
    merged = merge_segments(segs)
    return [s.to_dict() for s in merged]


@app.get("/sessions/{session_id}/export")
def get_export(session_id: str, fmt: str = "json"):
    segs = merge_segments(_store.get(session_id, []))
    if fmt == "json":
        return PlainTextResponse(export_json(segs), media_type="application/json")
    elif fmt == "txt":
        return PlainTextResponse(export_txt(segs))
    elif fmt == "srt":
        return PlainTextResponse(export_srt(segs))
    elif fmt == "vtt":
        return PlainTextResponse(export_vtt(segs))
    else:
        raise HTTPException(status_code=400, detail=f"Unknown format: {fmt}")
