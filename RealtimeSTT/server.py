import asyncio
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel

print("모델 로딩 중 (small)...")
model = WhisperModel("small", device="cpu", compute_type="int8")
print("준비 완료. http://localhost:8000 을 Chrome으로 여세요.\n")

app = FastAPI()


def do_transcribe(audio: np.ndarray) -> str:
    segments, _ = model.transcribe(
        audio,
        language="ko",
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.0,
        no_repeat_ngram_size=3,
        vad_filter=False,
    )
    return "".join(s.text for s in segments).strip()


@app.get("/")
async def index():
    return HTMLResponse(Path("index.html").read_text(encoding="utf-8"))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("브라우저 연결됨")
    current_id = None
    try:
        while True:
            msg = await ws.receive()
            if "text" in msg:
                current_id = json.loads(msg["text"]).get("id")
            elif "bytes" in msg:
                audio = np.frombuffer(msg["bytes"], dtype=np.float32).copy()
                text = await asyncio.to_thread(do_transcribe, audio)
                print(f">> {text or '(인식 없음)'}")
                await ws.send_text(json.dumps({"id": current_id, "text": text}))
    except Exception:
        print("브라우저 연결 종료")
