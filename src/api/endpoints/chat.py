from fastapi import APIRouter, HTTPException, Request, status
from src.models import ChatRequest, ChatResponse, ErrorResponse
from src.agents.gpt4o import GitHubGPTAgent
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
agent_service = GitHubGPTAgent()

@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"}
    },
    summary="Chat with AI",
    description="Process a chat conversation with the AI model"
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat conversation with the AI model.

    Args:
        request (ChatRequest): The chat request containing messages and optional parameters

    Returns:
        ChatResponse: The AI model's response

    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        logger.info("Processing chat request")
        messages = [msg.dict() for msg in request.messages]

        # Use default values if None is provided
        temperature = request.temperature if request.temperature is not None else 0.7
        max_tokens = request.max_tokens if request.max_tokens is not None else 4096

        logger.info(f"Processing chat with {len(messages)} messages")
        # Get the last message content
        user_message = messages[-1]["content"]
        
        # Use agent service to process the message
        result = await agent_service.run(user_message)

        logger.info("Successfully generated chat response")
        return ChatResponse(**result)
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