import math
import numpy as np
from faster_whisper import WhisperModel
from shared.schema import TranscriptSegment


class ASREngine:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
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
