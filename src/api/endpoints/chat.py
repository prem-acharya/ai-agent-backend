from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.agents.gpt4o import GPT4OAgent

router = APIRouter()

class Message(BaseModel):
    content: str

@router.post("/chat")
async def stream_chat(message: Message):
    agent = GPT4OAgent()
    return await agent.process_chat_request(message.content)
