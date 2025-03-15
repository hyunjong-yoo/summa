import asyncio
import json
import logging
import uuid
from typing import Dict
import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis.asyncio as redis
from aiohttp import ClientSession

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-5s][%(name)-20s][%(threadName)s]|%(message)s"
)
logger = logging.getLogger("APIServer")

# STT 서버 URL
STT_SERVER_URL = "http://localhost:8001"

class APIServer:
    def __init__(self):
        self.app = FastAPI(title="Summa API Server")
        
        # CORS 미들웨어 추가
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:8080"],  # 클라이언트 출처 허용
            allow_credentials=True,
            allow_methods=["*"],  # 모든 HTTP 메서드 허용 (OPTIONS 포함)
            allow_headers=["*"],  # 모든 헤더 허용
        )

        # Redis 연결
        self.redis_client = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            password="mypassword",
            decode_responses=True
        )
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.register_endpoints()

        @self.app.on_event("startup")
        async def startup_event():
            await self.test_redis()

    async def test_redis(self):
        try:
            await self.redis_client.ping()
            logger.info("[RedisTest] information. status=connected")
        except Exception as e:
            logger.error(f"[RedisTest] error. error_message={str(e)}")

    async def shutdown(self):
        logger.info("[APIServerShutdown] information. status=shutting_down")
        for session_id, ws in list(self.websocket_connections.items()):
            try:
                await ws.close()
                logger.info(f"[WebSocketClose] information. session_id={session_id},status=closed")
            except Exception as e:
                logger.error(f"[WebSocketClose] error. error_message={str(e)},session_id={session_id}")
            finally:
                del self.websocket_connections[session_id]
        await self.redis_client.close()
        logger.info("[APIServerShutdown] information. status=shutdown_complete")

    def register_endpoints(self):
        @self.app.websocket("/summa-api/v1/sessions/{session_id}/ws")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            logger.info(f"[WebSocketEndpoint] information. session_id={session_id}")
            await websocket.accept()
            self.websocket_connections[session_id] = websocket
            try:
                while True:
                    audio_chunk = await websocket.receive_bytes()
                    logger.info(f"[WebSocketReceive] information. session_id={session_id},chunk_size={len(audio_chunk)}")
                    async with ClientSession() as session:
                        async with session.post(
                            f"{STT_SERVER_URL}/summa-stt/v1/transcripts/real-time",
                            json={"session_id": session_id, "audio": audio_chunk.hex()}
                        ) as resp:
                            if resp.status == 200:
                                response = await resp.json()
                                logger.info(f"[stt_response] information. session_id={session_id}, response={response}")
                                text = response.get("text", "")
                                #await websocket.send_text(json.dumps({"text": text}))
                                await websocket.send_text(text)
                                logger.info(f"[ws_send] information. session_id={session_id}, text={text}")
                            else:
                                logger.error(f"[STTRequest] error. error_message=STTServerError,session_id={session_id},status_code={resp.status}")
            except WebSocketDisconnect:
                logger.info(f"[WebSocketEndpoint] information. session_id={session_id},status=disconnected")
                del self.websocket_connections[session_id]
            except Exception as e:
                logger.error(f"[WebSocketEndpoint] error. error_message={str(e)},session_id={session_id}")
                del self.websocket_connections[session_id]

        class SessionCreateRequest(BaseModel):
            user_id: str = "anonymous"

        @self.app.post("/summa-api/v1/sessions/create", response_model=dict)
        async def create_session(request: SessionCreateRequest):
            logger.info(f"[CreateSession] information. user_id={request.user_id}")
            session_id = str(uuid.uuid4())
            try:
                await self.redis_client.hset(f"session:{session_id}", mapping={
                    "user_id": request.user_id,
                    "status": "active",
                    "transcripts": json.dumps([])
                })
                logger.info(f"[CreateSession] information. session_id={session_id},status=created")
                return {"session_id": session_id, "message": "Session created"}
            except Exception as e:
                logger.error(f"[CreateSession] error. error_message={str(e)},user_id={request.user_id}")
                raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

        @self.app.post("/summa-api/v1/sessions/{session_id}/finalize", response_model=dict)
        async def finalize_session(session_id: str):
            logger.info(f"[FinalizeSession] information. session_id={session_id}")
            try:
                async with ClientSession() as session:
                    async with session.get(f"{STT_SERVER_URL}/summa-stt/v1/transcripts/{session_id}/full") as resp:
                        if resp.status == 200:
                            full_transcript = await resp.json()
                            await self.redis_client.hset(f"session:{session_id}", "transcripts", json.dumps(full_transcript["transcripts"]))
                            if session_id in self.websocket_connections:
                                await self.websocket_connections[session_id].send_text(json.dumps({"final_text": full_transcript["transcripts"]}))
                            logger.info(f"[FinalizeSession] information. session_id={session_id},transcript_count={len(full_transcript['transcripts'])}")
                            return full_transcript
                        else:
                            logger.error(f"[FinalizeSession] error. error_message=STTServerError,session_id={session_id},status_code={resp.status}")
                            raise HTTPException(status_code=resp.status, detail="STT server error")
            except Exception as e:
                logger.error(f"[FinalizeSession] error. error_message={str(e)},session_id={session_id}")
                raise HTTPException(status_code=500, detail=f"Failed to finalize session: {str(e)}")

        @self.app.get("/summa-api/v1/sessions/{session_id}/data", response_model=dict)
        async def get_session_data(session_id: str):
            logger.info(f"[GetSessionData] information. session_id={session_id}")
            try:
                session_data = await self.redis_client.hgetall(f"session:{session_id}")
                if not session_data:
                    logger.error(f"[GetSessionData] error. error_message=SessionNotFound,session_id={session_id}")
                    raise HTTPException(status_code=404, detail="Session not found")
                session_data["transcripts"] = json.loads(session_data.get("transcripts", "[]"))
                logger.info(f"[GetSessionData] information. session_id={session_id},status=retrieved")
                return session_data
            except Exception as e:
                logger.error(f"[GetSessionData] error. error_message={str(e)},session_id={session_id}")
                raise HTTPException(status_code=500, detail=f"Failed to get session data: {str(e)}")

        @self.app.get("/summa-api/v1/downloads/{session_id}/json")
        async def download_session_json(session_id: str):
            logger.info(f"[DownloadSessionJson] information. session_id={session_id}")
            try:
                session_data = await self.redis_client.hgetall(f"session:{session_id}")
                if not session_data:
                    logger.error(f"[DownloadSessionJson] error. error_message=SessionNotFound,session_id={session_id}")
                    raise HTTPException(status_code=404, detail="Session not found")
                session_data["transcripts"] = json.loads(session_data.get("transcripts", "[]"))
                logger.info(f"[DownloadSessionJson] information. session_id={session_id},status=downloaded")
                return session_data
            except Exception as e:
                logger.error(f"[DownloadSessionJson] error. error_message={str(e)},session_id={session_id}")
                raise HTTPException(status_code=500, detail=f"Failed to download session: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    server = APIServer()
    try:
        uvicorn.run(server.app, host="0.0.0.0", port=8000)
    finally:
        asyncio.run(server.shutdown())