from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from shared.schema import TranscriptSegment

if TYPE_CHECKING:
    from client.vad import VADEngine
    from client.asr import ASREngine


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
