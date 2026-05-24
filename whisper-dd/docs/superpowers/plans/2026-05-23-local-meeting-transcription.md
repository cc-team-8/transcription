# Local Meeting Transcription 구현 계획



> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 참가자별 디바이스에서 온디바이스 VAD + ASR을 수행하고, 로컬 허브에서 전사 결과를 시간순 병합해 회의록을 생성한다.

**Architecture:** 각 클라이언트 디바이스는 sounddevice로 캡처 → silero-vad로 발화 구간 검출 → faster-whisper로 전사 후, JSON 세그먼트를 로컬 허브(FastAPI)에 POST한다. 허브는 모든 참가자 결과를 start_time 기준으로 정렬·병합하여 JSON/SRT/VTT/TXT로 내보낸다.

**Tech Stack:**

- 오디오 캡처: `sounddevice`
- VAD: `silero-vad` (snakers4/silero-vad)
- ASR: `faster-whisper` (SYSTRAN/faster-whisper)
- 허브 서버: `FastAPI` + `uvicorn`
- 테스트: `pytest` + `httpx`

---

## 파일 구조

```
whisper-dd/
├── client/
│   ├── __init__.py
│   ├── capture.py        # sounddevice 실시간 캡처 → 큐
│   ├── normalizer.py     # 임의 오디오 → 16kHz mono float32
│   ├── vad.py            # silero-vad 래퍼 → 발화 구간 리스트
│   ├── asr.py            # faster-whisper 래퍼 → TranscriptSegment
│   └── pipeline.py       # 캡처→VAD→ASR→전송 오케스트레이터
├── hub/
│   ├── __init__.py
│   ├── merger.py         # 참가자별 세그먼트 시간순 병합
│   ├── exporter.py       # JSON/SRT/VTT/TXT 내보내기
│   └── server.py         # FastAPI 엔드포인트
├── shared/
│   ├── __init__.py
│   └── schema.py         # TranscriptSegment 공유 스키마
├── tests/
│   ├── test_normalizer.py
│   ├── test_vad.py
│   ├── test_asr.py
│   ├── test_merger.py
│   ├── test_exporter.py
│   └── test_server.py
├── requirements.txt
└── conftest.py
```

---

## Task 1: 프로젝트 셋업

**Files:**

- Create: `requirements.txt`
- Create: `shared/__init__.py`, `shared/schema.py`
- Create: `client/__init__.py`, `hub/__init__.py`
- Create: `conftest.py`

- [ ] **Step 1: requirements.txt 작성**

```text
sounddevice>=0.4.6
numpy>=1.24.0
silero-vad>=5.1.2
faster-whisper>=1.1.0
fastapi>=0.111.0
uvicorn>=0.30.0
httpx>=0.27.0
pytest>=8.0.0
```

- [ ] **Step 2: 패키지 설치**

```bash
pip install -r requirements.txt
```

Expected: 에러 없이 설치 완료.

- [ ] **Step 3: shared/schema.py 작성**

```python
from dataclasses import dataclass, asdict
import json


@dataclass
class TranscriptSegment:
    session_id: str
    participant_id: str
    start_time: float   # 세션 시작부터 초 단위
    end_time: float
    text: str
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptSegment":
        return cls(**d)
```

- [ ] **Step 4: 빈 **init**.py 생성**

```bash
touch shared/__init__.py client/__init__.py hub/__init__.py
```

- [ ] **Step 5: conftest.py 작성**

```python
import numpy as np
import pytest


@pytest.fixture
def sample_audio_16k():
    """16kHz mono float32 1초 무음 + 0.5초 tone + 0.5초 무음"""
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
```

- [ ] **Step 6: 커밋**

```bash
git add requirements.txt shared/ client/__init__.py hub/__init__.py conftest.py
git commit -m "feat: project scaffold with shared schema"
```

---

## Task 2: Audio Normalizer

**Files:**

- Create: `client/normalizer.py`
- Create: `tests/test_normalizer.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_normalizer.py
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
    # 44100→16000: 비율 허용 ±2%
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/test_normalizer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'client.normalizer'`

- [ ] **Step 3: normalizer.py 구현**

```python
# client/normalizer.py
import numpy as np


def normalize_audio(audio: np.ndarray, src_sr: int, target_sr: int = 16000) -> np.ndarray:
    """임의 오디오 배열을 target_sr mono float32로 변환한다."""
    # stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    # 리샘플링 (scipy 없이 선형 보간)
    if src_sr != target_sr:
        original_len = len(audio)
        target_len = int(original_len * target_sr / src_sr)
        indices = np.linspace(0, original_len - 1, target_len)
        audio = np.interp(indices, np.arange(original_len), audio).astype(np.float32)

    # 클리핑
    audio = np.clip(audio, -1.0, 1.0)
    return audio
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_normalizer.py -v
```

Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add client/normalizer.py tests/test_normalizer.py
git commit -m "feat: audio normalizer (resample + stereo-to-mono + clip)"
```

---

## Task 3: VAD Engine (silero-vad)

**Files:**

- Create: `client/vad.py`
- Create: `tests/test_vad.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_vad.py
import numpy as np
import pytest
from client.vad import VADEngine


@pytest.fixture(scope="module")
def vad():
    return VADEngine()


def test_detects_speech_in_tone(vad, sample_audio_16k):
    # sample_audio_16k = 1초 무음 + 0.5초 tone + 0.5초 무음
    segments = vad.get_speech_segments(sample_audio_16k, sr=16000)
    # 최소 1개 이상 발화 구간이 검출되어야 함
    assert len(segments) >= 1


def test_returns_sample_indices(vad, sample_audio_16k):
    segments = vad.get_speech_segments(sample_audio_16k, sr=16000)
    for start, end in segments:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert 0 <= start < end <= len(sample_audio_16k)


def test_silence_only_returns_no_segments(vad):
    silence = np.zeros(16000, dtype=np.float32)
    segments = vad.get_speech_segments(silence, sr=16000)
    assert len(segments) == 0
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/test_vad.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: vad.py 구현**

```python
# client/vad.py
import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps


class VADEngine:
    def __init__(self, threshold: float = 0.5):
        self.model = load_silero_vad()
        self.threshold = threshold

    def get_speech_segments(
        self, audio: np.ndarray, sr: int = 16000
    ) -> list[tuple[int, int]]:
        """발화 구간을 (start_sample, end_sample) 리스트로 반환한다."""
        tensor = torch.from_numpy(audio)
        timestamps = get_speech_timestamps(
            tensor,
            self.model,
            sampling_rate=sr,
            threshold=self.threshold,
            return_seconds=False,
        )
        return [(t["start"], t["end"]) for t in timestamps]
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_vad.py -v
```

Expected: 3 passed (모델 첫 로드 시 다운로드 발생할 수 있음)

- [ ] **Step 5: 커밋**

```bash
git add client/vad.py tests/test_vad.py
git commit -m "feat: silero-vad wrapper with segment detection"
```

---

## Task 4: ASR Engine (faster-whisper)

**Files:**

- Create: `client/asr.py`
- Create: `tests/test_asr.py`
- Create: `tests/fixtures/short_ko.wav` (테스트용 오디오, 별도 생성)

> **참고:** 테스트는 실제 모델 추론을 mock으로 대체해 빠르게 실행한다.

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_asr.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from shared.schema import TranscriptSegment
from client.asr import ASREngine


@pytest.fixture
def mock_asr():
    with patch("client.asr.WhisperModel") as MockModel:
        mock_instance = MagicMock()
        # faster-whisper segments는 generator로 반환됨
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 2.5
        mock_segment.text = " 안녕하세요"
        mock_segment.avg_logprob = -0.2  # confidence 계산용
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/test_asr.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: asr.py 구현**

```python
# client/asr.py
import math
import numpy as np
from faster_whisper import WhisperModel
from shared.schema import TranscriptSegment


class ASREngine:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio: np.ndarray,
        session_id: str,
        participant_id: str,
        time_offset: float = 0.0,
        language: str = "ko",
    ) -> list[TranscriptSegment]:
        """오디오 배열을 전사해 TranscriptSegment 리스트로 반환한다."""
        segments, _ = self.model.transcribe(audio, language=language, beam_size=5)
        result = []
        for seg in segments:
            confidence = min(1.0, max(0.0, math.exp(seg.avg_logprob)))
            result.append(
                TranscriptSegment(
                    session_id=session_id,
                    participant_id=participant_id,
                    start_time=round(seg.start + time_offset, 3),
                    end_time=round(seg.end + time_offset, 3),
                    text=seg.text.strip(),
                    confidence=round(confidence, 4),
                )
            )
        return result
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_asr.py -v
```

Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add client/asr.py tests/test_asr.py
git commit -m "feat: faster-whisper ASR wrapper with timestamp offset"
```

---

## Task 5: Client Pipeline

**Files:**

- Create: `client/pipeline.py`
- Create: `tests/test_pipeline.py`

파이프라인은 "청크 버퍼에 오디오 누적 → VAD → 발화 구간만 ASR → 세그먼트 출력" 루프다.

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_pipeline.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from client.pipeline import AudioPipeline
from shared.schema import TranscriptSegment


@pytest.fixture
def mock_pipeline():
    mock_vad = MagicMock()
    mock_asr = MagicMock()

    # VAD: 전체 오디오를 하나의 발화 구간으로 반환
    mock_vad.get_speech_segments.return_value = [(0, 16000)]

    # ASR: 더미 세그먼트 반환
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
    mock_vad.get_speech_segments.return_value = []  # 발화 없음
    audio = np.zeros(16000, dtype=np.float32)
    segments = pipeline.process_chunk(audio, time_offset=0.0)
    mock_asr.transcribe.assert_not_called()
    assert segments == []
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/test_pipeline.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: pipeline.py 구현**

```python
# client/pipeline.py
import numpy as np
from client.vad import VADEngine
from client.asr import ASREngine
from shared.schema import TranscriptSegment


class AudioPipeline:
    def __init__(
        self,
        vad: VADEngine,
        asr: ASREngine,
        session_id: str,
        participant_id: str,
        chunk_duration: float = 5.0,
        sr: int = 16000,
    ):
        self.vad = vad
        self.asr = asr
        self.session_id = session_id
        self.participant_id = participant_id
        self.sr = sr

    def process_chunk(
        self, audio: np.ndarray, time_offset: float
    ) -> list[TranscriptSegment]:
        """오디오 청크에서 발화 구간을 검출하고 전사해 반환한다."""
        speech_segments = self.vad.get_speech_segments(audio, sr=self.sr)
        if not speech_segments:
            return []

        results = []
        for start_sample, end_sample in speech_segments:
            segment_audio = audio[start_sample:end_sample]
            segment_offset = time_offset + start_sample / self.sr
            segs = self.asr.transcribe(
                segment_audio,
                session_id=self.session_id,
                participant_id=self.participant_id,
                time_offset=segment_offset,
            )
            results.extend(segs)
        return results
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_pipeline.py -v
```

Expected: 2 passed

- [ ] **Step 5: 커밋**

```bash
git add client/pipeline.py tests/test_pipeline.py
git commit -m "feat: client audio pipeline (vad → asr per segment)"
```

---

## Task 6: Hub Merger

**Files:**

- Create: `hub/merger.py`
- Create: `tests/test_merger.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_merger.py
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/test_merger.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: merger.py 구현**

```python
# hub/merger.py
from shared.schema import TranscriptSegment


def merge_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """참가자별 세그먼트를 start_time 기준으로 정렬·병합한다."""
    return sorted(segments, key=lambda s: s.start_time)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_merger.py -v
```

Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add hub/merger.py tests/test_merger.py
git commit -m "feat: hub merger sorts segments by start_time"
```

---

## Task 7: Exporter (JSON / TXT / SRT / VTT)

**Files:**

- Create: `hub/exporter.py`
- Create: `tests/test_exporter.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_exporter.py
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/test_exporter.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: exporter.py 구현**

```python
# hub/exporter.py
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
        blocks.append(f"{i}\n{start} --> {end}\n[{s.participant_id}] {s.text}\n")
    return "\n".join(blocks)


def export_vtt(segments: list[TranscriptSegment]) -> str:
    header = "WEBVTT\n\n"
    blocks = []
    for s in segments:
        start = _fmt_time_vtt(s.start_time)
        end = _fmt_time_vtt(s.end_time)
        blocks.append(f"{start} --> {end}\n[{s.participant_id}] {s.text}")
    return header + "\n\n".join(blocks)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_exporter.py -v
```

Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add hub/exporter.py tests/test_exporter.py
git commit -m "feat: exporter supports JSON, TXT, SRT, VTT formats"
```

---

## Task 8: Hub Server (FastAPI)

**Files:**

- Create: `hub/server.py`
- Create: `tests/test_server.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_server.py
import pytest
from fastapi.testclient import TestClient
from hub.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_post_segment_returns_201(client):
    payload = {
        "session_id": "s1",
        "participant_id": "p01",
        "start_time": 1.0,
        "end_time": 3.5,
        "text": "회의 시작합니다",
        "confidence": 0.95,
    }
    resp = client.post("/segments", json=payload)
    assert resp.status_code == 201


def test_get_merged_returns_sorted_list(client):
    # 나중 시작 시간의 세그먼트를 먼저 POST
    client.post("/segments", json={
        "session_id": "s1", "participant_id": "p02",
        "start_time": 10.0, "end_time": 12.0,
        "text": "나중 발화", "confidence": 0.9,
    })
    client.post("/segments", json={
        "session_id": "s1", "participant_id": "p01",
        "start_time": 2.0, "end_time": 4.0,
        "text": "이른 발화", "confidence": 0.9,
    })
    resp = client.get("/sessions/s1/transcript")
    assert resp.status_code == 200
    data = resp.json()
    starts = [s["start_time"] for s in data]
    assert starts == sorted(starts)


def test_export_srt(client):
    resp = client.get("/sessions/s1/export?fmt=srt")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "WEBVTT" not in resp.text
    assert "-->" in resp.text


def test_unknown_session_returns_empty(client):
    resp = client.get("/sessions/nonexistent/transcript")
    assert resp.status_code == 200
    assert resp.json() == []
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
pytest tests/test_server.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: server.py 구현**

```python
# hub/server.py
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
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_server.py -v
```

Expected: 4 passed

> **주의:** TestClient는 앱 인스턴스를 공유하므로 `_store`가 테스트 간 누적된다. 현재 테스트는 이를 고려해 작성됨.

- [ ] **Step 5: 커밋**

```bash
git add hub/server.py tests/test_server.py
git commit -m "feat: FastAPI hub with POST /segments and GET export"
```

---

## Task 9: 전체 테스트 실행

- [ ] **Step 1: 모든 테스트 실행**

```bash
pytest tests/ -v --tb=short
```

Expected: 전체 통과 (normalizer 4 + vad 3 + asr 3 + pipeline 2 + merger 4 + exporter 4 + server 4 = 24 passed)

- [ ] **Step 2: 최종 커밋**

```bash
git add .
git commit -m "feat: complete local meeting transcription system (client + hub)"
```

---

## 오픈소스 선택 정리

| 모듈          | 라이브러리       | 이유                                             |
| ------------- | ---------------- | ------------------------------------------------ |
| 오디오 캡처   | `sounddevice`    | 크로스플랫폼, 실시간 스트리밍 콜백               |
| 오디오 정규화 | `numpy` 내장     | scipy 의존성 없이 선형 보간으로 충분             |
| VAD           | `silero-vad`     | 경량(~1MB), 온디바이스, 16kHz 최적화             |
| ASR           | `faster-whisper` | CTranslate2 int8 양자화로 CPU에서도 실용적 속도  |
| 정렬(선택)    | `WhisperX`       | 단어 단위 타임스탬프 필요 시 faster-whisper 교체 |
| 허브 서버     | `FastAPI`        | 타입 안전 Pydantic 검증, 비동기 지원             |

## 선택 범위 (추후 구현)

- **실시간 자막 뷰**: FastAPI WebSocket + 브라우저 SSE로 `/sessions/{id}/stream` 엔드포인트 추가
- **요약/액션 아이템 추출**: 병합된 TXT를 LLM API에 POST하는 별도 서비스
- **재전사**: 저장된 오디오 파일을 `ASREngine.transcribe()`에 재입력

## 제외 범위

- 화자 diarization (pyannote.audio): 디바이스당 1인 조건에서 불필요
- 클라우드 ASR (OpenAI Whisper API): 프라이버시 원칙 위배
- 보이스 인증
