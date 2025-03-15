import logging
import queue
import threading
import asyncio
from fastapi import FastAPI
import numpy as np
import webrtcvad
import whisper
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)-5s][%(name)-20s][%(threadName)s]|%(message)s")
logger = logging.getLogger(__name__)

class STTServer:
    def __init__(self):
        self.app = FastAPI()
        self.model = whisper.load_model("base")
        self.vad = webrtcvad.Vad(1)
        self.audio_queue = queue.Queue(maxsize=1000)
        self.transcribe_queue = asyncio.Queue(maxsize=1000)
        self.session_frames = {}
        self.transcript_store = {}
        self.lock = threading.Lock()
        self.audio_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.transcribe_thread = threading.Thread(target=self.run_transcribe_loop, daemon=True)
        self.audio_thread.start()
        self.transcribe_thread.start()
        self.setup_routes()

    def process_audio(self):
        while True:
            session_id, pcm_data = self.audio_queue.get()
            frame_size = 480  # 30ms at 16kHz
            frames = [pcm_data[i:i + frame_size * 2] for i in range(0, len(pcm_data), frame_size * 2)]
            with self.lock:
                if session_id not in self.session_frames:
                    self.session_frames[session_id] = []
                for frame in frames:
                    if len(frame) == frame_size * 2 and self.vad.is_speech(frame, 16000):
                        self.session_frames[session_id].append(np.frombuffer(frame, dtype=np.int16))
                        if len(self.session_frames[session_id]) >= 10:  # MIN_FRAMES_TO_TRANSCRIBE
                            asyncio.run_coroutine_threadsafe(
                                self.transcribe_queue.put((session_id, self.session_frames[session_id].copy())),
                                asyncio.get_event_loop()
                            )
                            self.session_frames[session_id] = []

    def run_transcribe_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.transcribe_audio())

    async def transcribe_audio(self):
        while True:
            session_id, frames = await self.transcribe_queue.get()
            audio = np.concatenate(frames).astype(np.float32) / 32768.0
            result = self.model.transcribe(audio, language="en")
            with self.lock:
                self.transcript_store[session_id] = result["text"]
            logger.info(f"[transcribe] information. session_id={session_id}, text={result['text']}")

    def setup_routes(self):
        @self.app.post("/summa-stt/v1/transcripts/real-time")
        async def real_time_transcript(data: dict):
            session_id = data["session_id"]
            pcm_data = bytes.fromhex(data["audio"])
            self.audio_queue.put((session_id, pcm_data))
            with self.lock:
                return self.transcript_store.get(session_id, "")

        @self.app.get("/summa-stt/v1/transcripts/{session_id}/full")
        async def full_transcript(session_id: str):
            with self.lock:
                text = self.transcript_store.get(session_id, "")
                if session_id in self.session_frames and self.session_frames[session_id]:
                    audio = np.concatenate(self.session_frames[session_id]).astype(np.float32) / 32768.0
                    result = self.model.transcribe(audio, language="en")
                    text += " " + result["text"]
                timestamp = datetime.utcnow().isoformat() + "Z"
                del self.session_frames[session_id]
                del self.transcript_store[session_id]
            logger.info(f"[full_transcript] information. session_id={session_id}")
            return {"timestamp": timestamp, "text": text}

app = STTServer().app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)