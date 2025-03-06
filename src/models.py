from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    role: Role = Field(..., description="The role of the message sender")
    content: str = Field(..., min_length=1, description="The content of the message")

    @validator("content")
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()

class AgentType(str, Enum):
    GEMINI = "gemini"
    GPT4O = "gpt4o"

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_items=1, description="List of messages in the conversation")
    websearch: Optional[bool] = Field(False, description="Enable web search")
    reasoning: Optional[bool] = Field(False, description="Enable chain-of-thought reasoning")
    model: Optional[AgentType] = Field(AgentType.GEMINI, description="Type of AI agent to use")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The response from the AI model")
    error: Optional[str] = Field(None, description="Error message if any")
    agent: str = Field("OpenAIAgent", description="The agent used for processing")
    model: str = Field("gpt-4o", description="The AI model used")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error detail message")
    code: str = Field(..., description="Error code for categorization")