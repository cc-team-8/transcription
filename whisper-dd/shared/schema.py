from dataclasses import dataclass, asdict


@dataclass
class TranscriptSegment:
    session_id: str
    participant_id: str
    start_time: float   # seconds from session start
    end_time: float
    text: str
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptSegment":
        return cls(**d)
