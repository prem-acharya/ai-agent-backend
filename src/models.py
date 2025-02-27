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

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_items=1, description="List of messages in the conversation")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperature for response generation")
    max_tokens: Optional[int] = Field(4096, ge=1, le=8192, description="Maximum tokens in the response")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The response from the AI model")
    error: Optional[str] = Field(None, description="Error message if any")
    agent: str = Field("ZeroShotAgent", description="The agent used for processing")
    model: str = Field("gpt-4o", description="The AI model used")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error detail message")
    code: str = Field(..., description="Error code for categorization")