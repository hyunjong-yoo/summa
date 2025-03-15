import logging
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis
import aiohttp
import uuid
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)-5s][%(name)-20s][%(threadName)s]|%(message)s")
logger = logging.getLogger(__name__)

class APIServer:
    def __init__(self):
        self.app = FastAPI()
        self.redis_client = redis.Redis(host='localhost', port=6379, password='mypassword', decode_responses=True)
        self.websocket_connections = {}
        self.app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8080"], allow_methods=["*"], allow_headers=["*"])
        self.setup_routes()

    def setup_routes(self):
        @self.app.websocket("/summa-api/v1/sessions/{session_id}/ws")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            await websocket.accept()
            self.websocket_connections[session_id] = websocket
            logger.info(f"[websocket_connect] information. session_id={session_id}")
            try:
                while True:
                    pcm_data = await websocket.receive_bytes()
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                        async with session.post('http://localhost:8001/summa-stt/v1/transcripts/real-time', 
                                                json={"session_id": session_id, "audio": pcm_data.hex()}) as resp:
                            text = await resp.text()
                            await websocket.send_text(text)
                            await self.redis_client.hset(f"session:{session_id}", "transcripts", text)
            except Exception as e:
                logger.error(f"[websocket] error. error_message={str(e)}, session_id={session_id}")
            finally:
                del self.websocket_connections[session_id]
                await self.redis_client.delete(f"session:{session_id}")

        @self.app.post("/summa-api/v1/sessions/create")
        async def create_session(user_id: str):
            session_id = str(uuid.uuid4())
            await self.redis_client.hset(f"session:{session_id}", mapping={"user_id": user_id, "status": "active"})
            logger.info(f"[create_session] information. session_id={session_id}, user_id={user_id}")
            return {"session_id": session_id}

        @self.app.post("/summa-api/v1/sessions/{session_id}/finalize")
        async def finalize_session(session_id: str):
            text = await self.redis_client.hget(f"session:{session_id}", "transcripts") or ""
            await self.redis_client.hset(f"session:{session_id}", "status", "completed")
            logger.info(f"[finalize_session] information. session_id={session_id}")
            return {"timestamp": "2025-03-15T00:00:00Z", "text": text}

app = APIServer().app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)