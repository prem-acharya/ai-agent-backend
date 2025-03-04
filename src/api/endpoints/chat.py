from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from src.models import ChatRequest, ErrorResponse
from src.agents.gpt4o import GitHubGPTAgent
import logging
import json
import asyncio
from typing import AsyncGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
agent_service = GitHubGPTAgent()

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
        logger.info("Processing chat request")

        # extract messages from request
        messages = [msg.dict() for msg in request.messages]
        if not messages:
            raise ValueError("No messages provided in request")

        user_message = messages[-1]["content"]
        websearch = request.websearch if request.websearch is not None else False

        logger.info(f"User message: {user_message}, Web Search: {websearch}")
        
        # Get streaming response generator
        stream = await agent_service.run(user_message, websearch)
        
        # Return streaming response with proper content type
        return StreamingResponse(
            stream,
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": str(e), "code": "VALIDATION_ERROR"}
        )
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Internal server error", "code": "SERVER_ERROR"}
        )
    
async def stream_chat_response(query: str, websearch: bool) -> AsyncGenerator[str, None]:
    """Stream AI-generated responses as JSON chunks."""
    result = await agent_service.run(query, websearch)

    if isinstance(result, dict) and "response" in result:
        response_text = result["response"]
        words = response_text.split()
        for word in words:
            chunk = json.dumps({"chunk": word})  # Stream each word as JSON
            yield f"{chunk}\n"
            await asyncio.sleep(0.1)