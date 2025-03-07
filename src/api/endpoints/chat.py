from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncIterable
import asyncio
import os
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel

router = APIRouter()

class Message(BaseModel):
    content: str

async def generate_chat_response(content: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    model = AzureChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name="gpt-4o",
        model_name="gpt-4o",
        api_version="2024-02-15-preview",
        streaming=True,
        verbose=True,
        temperature=0.7,
        max_tokens=4096,
        callbacks=[callback]
    )

    task = asyncio.create_task(
        model.agenerate(messages=[[HumanMessage(content=content)]])
    )

    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        callback.done.set()

    await task

@router.post("/chat")
async def stream_chat(message: Message):
    try:
        generator = generate_chat_response(message.content)
        return StreamingResponse(
            generator, 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
