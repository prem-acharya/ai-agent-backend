from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from src.models import ChatRequest, ErrorResponse, AgentType
from src.agents.gemini import GeminiAgent
from src.agents.gpt4o import GitHubGPTAgent
import logging
import json
import asyncio
from typing import AsyncGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
gemini_agent = GeminiAgent()
gpt4o_agent = GitHubGPTAgent()

@router.post(
    "/chat",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"}
    },
    summary="Chat with AI agent",
    description="Process a chat conversation with the AI agent"
)
async def chat(request: ChatRequest):
    """
    Process a chat conversation with the AI agent.

    Args:
        request (ChatRequest): The chat request containing messages and optional parameters

    Returns:
        StreamingResponse: The AI model's response in JSON format.

    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        logger.info(f"Processing chat request with agent: {request.agent_type}")
        
        if not request.messages:
            raise ValueError("No messages provided in request")

        user_message = request.messages[-1].content
        
        # Select agent based on type
        if request.agent_type == AgentType.GEMINI:
            logger.info(f"Using Gemini agent with reasoning: {request.use_reasoning}")
            stream = gemini_agent.run(
                query=user_message,
                use_reasoning=request.use_reasoning or False
            )
        else:
            logger.info("Using GPT4O agent")
            stream = gpt4o_agent.run(
                query=user_message,
                websearch=request.websearch or False
            )

        return StreamingResponse(
            stream,
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(e), "code": "SERVER_ERROR"}
        )
    
async def stream_chat_response(query: str, websearch: bool) -> AsyncGenerator[str, None]:
    """Stream AI-generated responses as JSON chunks."""
    result = await gpt4o_agent.run(query, websearch)

    if isinstance(result, dict) and "response" in result:
        response_text = result["response"]
        words = response_text.split()
        for word in words:
            chunk = json.dumps({"chunk": word})  # Stream each word as JSON
            yield f"{chunk}\n"
            await asyncio.sleep(0.1)