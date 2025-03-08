from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.agents.gpt4o import GPT4OAgent
from typing import Optional

router = APIRouter()

class ChatRequest(BaseModel):
    content: str
    websearch: Optional[bool] = False

@router.post("/chat")
async def stream_chat(request: ChatRequest):
    agent = GPT4OAgent()
    return await agent.process_chat_request(
        content=request.content,
        websearch=request.websearch
    )
