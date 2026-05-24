import sys
from datetime import datetime
from RealtimeSTT import AudioToTextRecorder

start_time = None


def ts(dt=None):
    return (dt or datetime.now()).strftime("%H:%M:%S")


def on_recording_start():
    global start_time
    start_time = datetime.now()
    sys.stdout.write(f"\r[{ts(start_time)}] 🎤 듣는 중...          \n")
    sys.stdout.flush()


def on_recording_stop():
    sys.stdout.write("\r⏳ 처리 중...                    \n")
    sys.stdout.flush()


def on_realtime_text(text):
    sys.stdout.write(f"\r    {text[:80]}")
    sys.stdout.flush()


def on_final_text(text):
    end_time = datetime.now()
    text = text.strip()
    sys.stdout.write("\r" + " " * 84 + "\r")
    if text:
        print(f"[{ts(start_time)} ~ {ts(end_time)}] {text}\n")


if __name__ == "__main__":
    recorder = AudioToTextRecorder(
        language="ko",
        model="small",
        silero_sensitivity=0.4,
        silero_use_onnx=True,
        post_speech_silence_duration=0.7,
        on_recording_start=on_recording_start,
        on_recording_stop=on_recording_stop,
        on_realtime_transcription_update=on_realtime_text,
        enable_realtime_transcription=True,
        realtime_model_type="tiny",
        realtime_processing_pause=0.1,
    )

    print("말하세요... (Ctrl+C 종료)\n")
    try:
        while True:
            recorder.text(on_final_text)
    except KeyboardInterrupt:
        recorder.stop()
        print("\n종료합니다.")
