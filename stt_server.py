import asyncio
import json
import logging
import threading
import queue
from typing import Dict, List
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import webrtcvad
import whisper
import torch
import os
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)-20s][%(threadName)s]|%(message)s"
)
logger = logging.getLogger("STTServer")

# 설정값
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 샘플
CHUNK_SIZE = FRAME_SIZE * 2  # 960 바이트 (16비트 샘플)
SILENCE_THRESHOLD = 5
MIN_FRAMES_TO_TRANSCRIBE = 10
WHISPER_MODEL_NAME = "base"

class STTServer:
    def __init__(self):
        self.app = FastAPI(title="Summa STT Server")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ensure_model_downloaded(WHISPER_MODEL_NAME)
        self.model = whisper.load_model(
            WHISPER_MODEL_NAME,
            device=device
            #, compute_type="float32"
        )
        logger.info(f"[WhisperModel] information. model={WHISPER_MODEL_NAME},device={device},status=loaded")

        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)

        self.audio_queue = queue.Queue(maxsize=1000)
        self.transcribe_queue = asyncio.Queue(maxsize=1000)
        self.session_frames: Dict[str, List[bytes]] = {}
        self.transcript_store: Dict[str, List[dict]] = {}
        self.shutdown_event = threading.Event()
        self.lock = threading.Lock()

        self.audio_thread = threading.Thread(target=self.process_audio_thread, name="AudioProcessor")
        self.transcribe_thread = threading.Thread(target=self.transcribe_thread, name="Transcriber")
        self.audio_thread.start()
        self.transcribe_thread.start()

        self.register_endpoints()

    def ensure_model_downloaded(self, model_name: str):
        logger.info(f"[EnsureModelDownload] information. model={model_name},status=checking")
        cache_dir = Path.home() / ".cache" / "whisper"
        model_path = cache_dir / f"{model_name}.pt"

        if not model_path.exists():
            logger.info(f"[EnsureModelDownload] information. model={model_name},status=downloading")
            try:
                whisper.load_model(model_name, download_root=str(cache_dir))
                logger.info(f"[EnsureModelDownload] information. model={model_name},status=downloaded")
            except Exception as e:
                logger.error(f"[EnsureModelDownload] error. error_message={str(e)},model={model_name}")
                raise RuntimeError(f"Failed to download Whisper model {model_name}: {str(e)}")
        else:
            logger.info(f"[EnsureModelDownload] information. model={model_name},status=already_downloaded")

    async def shutdown(self):
        logger.info("[STTServerShutdown] information. status=shutting_down")
        self.shutdown_event.set()

        while not self.audio_queue.empty():
            try:
                session_id, audio_chunk = self.audio_queue.get_nowait()
                self.process_audio_chunk(session_id, audio_chunk)
                self.audio_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"[STTServerShutdown] error. error_message={str(e)},session_id={session_id}")

        with self.lock:
            for session_id, frames in list(self.session_frames.items()):
                if frames:
                    asyncio.run_coroutine_threadsafe(self.transcribe_queue.put((session_id, frames)), asyncio.get_event_loop())
            self.session_frames.clear()

        while not self.transcribe_queue.empty():
            try:
                session_id, frames = await self.transcribe_queue.get()
                await self.process_transcribe(session_id, frames)
                self.transcribe_queue.task_done()
            except Exception as e:
                logger.error(f"[STTServerShutdown] error. error_message={str(e)},session_id={session_id}")

        self.audio_thread.join()
        self.transcribe_thread.join()
        logger.info("[STTServerShutdown] information. status=shutdown_complete")

    def process_audio_thread(self):
        logger.info("[ProcessAudioThread] information. status=started")
        while not self.shutdown_event.is_set():
            try:
                session_id, audio_chunk = self.audio_queue.get(timeout=1.0)
                self.process_audio_chunk(session_id, audio_chunk)
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[ProcessAudioThread] error. error_message={str(e)},session_id={session_id}")
        logger.info("[ProcessAudioThread] information. status=stopped")

    def process_audio_chunk(self, session_id: str, audio_chunk: bytes):
        logger.info(f"[ProcessAudioChunk] information. session_id={session_id},chunk_size={len(audio_chunk)}")
        try:
            # 청크 크기를 2의 배수로 조정 (np.int16 = 2바이트)
            chunk_size = len(audio_chunk)
            if chunk_size % 2 != 0:
                # 홀수 바이트면 마지막 바이트 제거
                audio_chunk = audio_chunk[:-1]
                logger.warning(f"[ProcessAudioChunk] information. session_id={session_id},original_chunk_size={chunk_size},adjusted_chunk_size={len(audio_chunk)}")

            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            frames = []
            for i in range(0, len(audio_data), FRAME_SIZE):
                frame = audio_data[i:i + FRAME_SIZE]
                if len(frame) == FRAME_SIZE:
                    frame_bytes = frame.tobytes()
                    frames.append(frame_bytes)

            with self.lock:
                if session_id not in self.session_frames:
                    self.session_frames[session_id] = []
                self.session_frames[session_id].extend(frames)
            self.check_silence_and_transcribe(session_id)
        except Exception as e:
            logger.error(f"[ProcessAudioChunk] error. error_message={str(e)},session_id={session_id},chunk_size={len(audio_chunk)}")

    def check_silence_and_transcribe(self, session_id: str):
        with self.lock:
            frames = self.session_frames.get(session_id, [])
            if not frames:
                return
            silence_count = 0
            for frame in frames[-SILENCE_THRESHOLD:]:
                if not self.vad.is_speech(frame, SAMPLE_RATE):
                    silence_count += 1

            if silence_count >= SILENCE_THRESHOLD and len(frames) >= MIN_FRAMES_TO_TRANSCRIBE:
                logger.info(f"[CheckSilence] information. session_id={session_id},frame_count={len(frames)},silence_detected=true")
                asyncio.run_coroutine_threadsafe(self.transcribe_queue.put((session_id, frames[:])), asyncio.get_event_loop())
                self.session_frames[session_id] = frames[-SILENCE_THRESHOLD:]

    def transcribe_thread(self):
        logger.info("[TranscribeThread] information. status=started")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def process():
            while not self.shutdown_event.is_set():
                try:
                    session_id, frames = await self.transcribe_queue.get()
                    await self.process_transcribe(session_id, frames)
                    self.transcribe_queue.task_done()
                except Exception as e:
                    logger.error(f"[TranscribeThread] error. error_message={str(e)},session_id={session_id}")

        loop.run_until_complete(process())
        logger.info("[TranscribeThread] information. status=stopped")

    async def process_transcribe(self, session_id: str, frames: List[bytes]):
        logger.info(f"[ProcessTranscribe] information. session_id={session_id},frame_count={len(frames)}")
        try:
            combined_audio = b"".join(frames)
            audio_np = np.frombuffer(combined_audio, dtype=np.int16).astype(np.float32) / 32768.0
            result = self.model.transcribe(audio_np, language="ko")
            text = result["text"]
            with self.lock:
                if session_id not in self.transcript_store:
                    self.transcript_store[session_id] = []
                self.transcript_store[session_id].append({"text": text, "timestamp": "2025-03-14T00:00:00Z"})
            logger.info(f"[ProcessTranscribe] information. session_id={session_id},text_length={len(text)}")
        except Exception as e:
            logger.error(f"[ProcessTranscribe] error. error_message={str(e)},session_id={session_id},frame_count={len(frames)}")

    def register_endpoints(self):
        class RealTimeTranscriptRequest(BaseModel):
            session_id: str
            audio: str  # Hex-encoded bytes

        @self.app.post("/summa-stt/v1/transcripts/real-time")
        async def real_time_transcript(data: dict):
            session_id = data["session_id"]
            pcm_data = bytes.fromhex(data["audio"])
            frame_size = 480  # 30ms at 16kHz
            frames = [pcm_data[i:i + frame_size * 2] for i in range(0, len(pcm_data), frame_size * 2)]
            
            with self.lock:
                if session_id not in self.session_frames:
                    self.session_frames[session_id] = []
                for frame in frames:
                    if len(frame) == frame_size * 2 and self.vad.is_speech(frame, 16000):
                        self.session_frames[session_id].append(np.frombuffer(frame, dtype=np.int16))
                
                if len(self.session_frames[session_id]) >= 10:  # MIN_FRAMES_TO_TRANSCRIBE
                    audio = np.concatenate(self.session_frames[session_id]).astype(np.float32) / 32768.0
                    result = self.model.transcribe(audio, language="en")
                    self.transcript_store[session_id] = result["text"]
                    self.session_frames[session_id] = []
                    logger.info(f"[real_time_transcript] information. session_id={session_id}, text={result['text']}")
                    return result["text"]
            
            return self.transcript_store.get(session_id, "")

        @self.app.get("/summa-stt/v1/transcripts/{session_id}/full", response_model=dict)
        async def get_full_transcript(session_id: str):
            logger.info(f"[GetFullTranscript] information. session_id={session_id}")
            if session_id not in self.transcript_store:
                logger.error(f"[GetFullTranscript] error. error_message=SessionNotFound,session_id={session_id}")
                raise HTTPException(status_code=404, detail="Session not found")

            try:
                with self.lock:
                    if session_id in self.session_frames and self.session_frames[session_id]:
                        frames = self.session_frames[session_id]
                        await self.process_transcribe(session_id, frames)
                    transcripts = self.transcript_store[session_id]
                    if session_id in self.session_frames:
                        del self.session_frames[session_id]
                    if session_id in self.transcript_store:
                        del self.transcript_store[session_id]
                logger.info(f"[GetFullTranscript] information. session_id={session_id},transcript_count={len(transcripts)}")
                return {"session_id": session_id, "transcripts": transcripts}
            except Exception as e:
                logger.error(f"[GetFullTranscript] error. error_message={str(e)},session_id={session_id}")
                raise HTTPException(status_code=500, detail=f"Failed to get full transcript: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    server = STTServer()
    try:
        uvicorn.run(server.app, host="0.0.0.0", port=8001)
    finally:
        asyncio.run(server.shutdown())