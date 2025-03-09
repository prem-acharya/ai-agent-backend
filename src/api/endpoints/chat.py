from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.agents.gpt4o import GPT4OAgent
from src.agents.gemini import GeminiAgent
from typing import Optional, Literal

router = APIRouter()

class ChatRequest(BaseModel):
    content: str
    model: Literal["gpt4o", "gemini"] = "gpt4"
    websearch: Optional[bool] = False
    reasoning: Optional[bool] = False

@router.post("/chat")
async def stream_chat(request: ChatRequest):
    if request.model == "gpt4o":
        agent = GPT4OAgent(websearch=request.websearch, reasoning=request.reasoning)
    else:
        agent = GeminiAgent(websearch=request.websearch, reasoning=request.reasoning)
        
    return await agent.process_chat_request(
        content=request.content,
        websearch=request.websearch,
        reasoning=request.reasoning
    )
