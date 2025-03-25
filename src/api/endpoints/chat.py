from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.agents.gpt4o import GPT4OAgent
from src.agents.gemini import GeminiAgent
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    content: str
    model: Literal["gpt4o", "gemini"] = "gpt4"
    websearch: Optional[bool] = False
    reasoning: Optional[bool] = False
    google_access_token: Optional[str] = None

@router.post("/chat")
async def stream_chat(request: ChatRequest):
    # Log if Google access token is provided (without revealing the token itself)
    if request.google_access_token:
        token_preview = request.google_access_token[:10] + "..." if request.google_access_token else "None"
        logger.info(f"Google access token provided: {token_preview}")
        logger.info(f"Google access token length: {len(request.google_access_token) if request.google_access_token else 0}")
    else:
        logger.warning("No Google access token provided in request")
    
    try:
        if request.model == "gpt4o":
            # For GPT4O, don't pass the Google access token as it's not currently supported
            logger.info(f"Initializing GPT4O agent. Note: Google Tasks integration not available for this model.")
            agent = GPT4OAgent(websearch=request.websearch, reasoning=request.reasoning)
        else:
            # For Gemini, pass the Google access token as it's supported
            logger.info(f"Initializing Gemini agent with Google Tasks access: {bool(request.google_access_token)}")
            agent = GeminiAgent(websearch=request.websearch, reasoning=request.reasoning, google_access_token=request.google_access_token)
            
        return await agent.process_chat_request(
            content=request.content,
            websearch=request.websearch,
            reasoning=request.reasoning
        )
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error initializing agent: {str(e)}")
