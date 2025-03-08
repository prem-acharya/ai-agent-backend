from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncIterable
from langchain.schema import HumanMessage
import asyncio
from src.utils.streaming import BaseStreamingLLM

class GPT4OAgent(BaseStreamingLLM):
    """GPT-4 Optimized Agent with streaming capabilities"""

    async def generate_response(self, content: str) -> AsyncIterable[str]:
        """Generate streaming response from the model"""
        try:
            messages = [HumanMessage(content=content)]
            
            task = asyncio.create_task(
                self.llm.agenerate(messages=[messages])
            )

            async for token in self.generate_streaming_response(messages):
                yield token

            await task

        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Generation error: {str(e)}"
            )
        finally:
            self.callback.done.set()

    async def process_chat_request(self, content: str) -> StreamingResponse:
        """Process chat request and return streaming response"""
        try:
            return StreamingResponse(
                self.generate_response(content),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Request processing error: {str(e)}"
            )