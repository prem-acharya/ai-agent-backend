from fastapi import APIRouter, HTTPException, Request, status
from src.models import ChatRequest, ChatResponse, ErrorResponse
from src.agents.gpt4o import GitHubGPTAgent
import logging

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

        # extract messages from request
        messages = [msg.dict() for msg in request.messages]
        if not messages:
            raise ValueError("No messages provided in request")

        user_message = messages[-1]["content"]
        websearch = request.websearch if request.websearch is not None else False

        logger.info(f"User message: {user_message}, Web Search: {websearch}")
        
        # call AI agent
        result = await agent_service.run(user_message, websearch)

        # check if result is a dictionary before unpacking
        if not isinstance(result, dict):
            raise ValueError("Invalid response format from AI agent")

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