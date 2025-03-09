import os
import asyncio
from functools import lru_cache
from typing import AsyncIterable

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import logging

logger = logging.getLogger(__name__)

class BaseGeminiStreaming:
    def __init__(self):
        self.callback = AsyncIteratorCallbackHandler()
        self.llm = self._initialize_llm()

    @lru_cache(maxsize=1)
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """
        Initialize Gemini with streaming enabled.
        Make sure GEMINI_API_KEY is set in your environment.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"),
            streaming=True,
            verbose=True,
            temperature=float(os.getenv("MODEL_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MODEL_MAX_TOKENS", "4096")),
            callbacks=[self.callback]
        )

    async def reset_callback(self):
        """Reset the callback handler for a fresh streaming session."""
        if hasattr(self.callback, "done") and not self.callback.done.is_set():
            self.callback.done.set()
            await asyncio.sleep(0.1)  # Allow for cleanup
        self.callback = AsyncIteratorCallbackHandler()
        self.llm.callbacks = [self.callback]
        logger.info("Callback handler reset.")

    async def stream_tokens(self) -> AsyncIterable[str]:
        """
        Stream tokens token-by-token from the callback handler.
        A fixed delay of 0.1 seconds is added after each token.
        """
        try:
            async for token in self.callback.aiter():
                text = token.get("text", "") if isinstance(token, dict) else token
                if text:
                    yield text
                    await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error streaming tokens: {str(e)}")
            yield f"\nStreaming error: {str(e)}"
        finally:
            if not self.callback.done.is_set():
                await self.callback.done.wait()

    async def generate_streaming_response(
        self, 
        content: str, 
        system_message: str = None
    ) -> AsyncIterable[str]:
        """
        Generate a streaming response by sending messages to the Gemini LLM.
        This method yields tokens one-by-one.
        """
        try:
            await self.reset_callback()
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=content))
            logger.info(f"Starting generation for: {content}...")
            # Begin generation (this starts streaming tokens via the callback)
            await self.llm.agenerate([messages])
            async for token in self.stream_tokens():
                yield token
        except Exception as e:
            logger.error(f"Error generating streaming response: {str(e)}")
            yield f"\nError generating response: {str(e)}"
