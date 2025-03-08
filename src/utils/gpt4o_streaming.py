from typing import AsyncIterable, List
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_openai import AzureChatOpenAI
from langchain.schema import BaseMessage
import os
from functools import lru_cache
import asyncio

class BaseStreamingLLM:
    def __init__(self):
        self.callback = AsyncIteratorCallbackHandler()
        self.llm = self._initialize_llm()

    @lru_cache(maxsize=1)
    def _initialize_llm(self) -> AzureChatOpenAI:
        """Initialize Azure ChatOpenAI with cached instance"""
        return AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o"),
            model_name=os.getenv("AZURE_MODEL_NAME", "gpt-4o"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
            streaming=True,
            verbose=True,
            temperature=float(os.getenv("MODEL_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MODEL_MAX_TOKENS", "4096")),
            callbacks=[self.callback]
        )

    async def stream_tokens(self) -> AsyncIterable[str]:
        """Stream tokens from the callback handler"""
        try:
            async for token in self.callback.aiter():
                yield token
        except Exception as e:
            yield f"Error: {str(e)}"
        finally:
            await self.callback.done.wait()

    async def generate_streaming_response(
        self, 
        messages: List[BaseMessage]
    ) -> AsyncIterable[str]:
        """Generate streaming response from messages"""
        try:
            await self.llm.agenerate([messages])
            async for token in self.stream_tokens():
                yield token
        except Exception as e:
            yield f"Error: {str(e)}" 