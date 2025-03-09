import os
import asyncio
from functools import lru_cache
from typing import AsyncIterable
import logging

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from langchain.prompts import PromptTemplate
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

from src.tools.datetime.time_tool import CurrentTimeTool
from src.tools.websearch.websearch_tool import WebSearchTool
from src.utils.gemini_streaming import BaseGeminiStreaming
from src.utils.prompts import initialize_prompts

logger = logging.getLogger(__name__)

# Define a custom pass-through class since RunnablePassthrough is not available
class RunnablePassthrough:
    def __call__(self, x):
        return x

class GeminiAgent(BaseGeminiStreaming):
    """Gemini Agent with token-by-token streaming capabilities and optional web search and reasoning."""

    def __init__(self, websearch: bool = False, reasoning: bool = False):
        super().__init__()
        self.websearch = websearch
        self.reasoning = reasoning
        # Load prompt templates (chain-of-thought, direct, final)
        self.cot_prompt, self.direct_prompt, self.final_prompt = initialize_prompts()
        self._initialize_chains()
        if websearch:
            self.tools = [CurrentTimeTool(), WebSearchTool()]
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )

    @lru_cache(maxsize=2)
    def _initialize_chains(self):
        """Initialize chain pipelines using a custom pass-through."""
        self.cot_chain = (
            RunnablePassthrough() | 
            self.cot_prompt | 
            self.llm | 
            (lambda x: {"text": x.content})
        )
        self.direct_chain = (
            RunnablePassthrough() | 
            self.direct_prompt | 
            self.llm | 
            (lambda x: {"text": x.content})
        )
        self.final_chain = (
            RunnablePassthrough() | 
            self.final_prompt | 
            self.llm | 
            (lambda x: {"text": x.content})
        )

    async def _reset_callback(self):
        """Reset the asynchronous callback handler for fresh streaming."""
        if not self.llm.callbacks[0].done.is_set():
            self.llm.callbacks[0].done.set()
        await asyncio.sleep(0.1)

    async def generate_response(self, content: str) -> AsyncIterable[str]:
        """Generate a token-by-token streaming response from the model."""
        try:
            await self._reset_callback()
            web_results = ""
            # If web search is enabled, get current time and perform web search.
            if self.websearch:
                time_tool = CurrentTimeTool()
                current_time = await time_tool._arun()
                logger.info(f"Current Time: {current_time}")
                yield "🌐 Searching on web...\n\n"
                web_tool = WebSearchTool()
                web_results = await web_tool._arun(content)
            if self.reasoning:
                yield "Reasoning:\n\n"
                await asyncio.sleep(0.1)
                # Generate chain-of-thought reasoning using the combined question.
                cot_result = await self.cot_chain.ainvoke({"question": content})
                yield cot_result["text"]
                await asyncio.sleep(0.1)
                await self._reset_callback()
                yield "\n\nFinal Answer:\n\n"
                await asyncio.sleep(0.2)
                final_result = await self.final_chain.ainvoke({
                    "chain_of_thought": cot_result["text"],
                    "web_context": web_results
                })
                yield final_result["text"]
                await asyncio.sleep(0.2)
            else:
                try:
                    direct_result = await self.direct_chain.ainvoke({"question": content})
                    yield direct_result["text"]
                    await asyncio.sleep(0.2)
                except Exception as e:
                    yield f"\nError in direct response: {str(e)}\n"
                    await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Gemini agent error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Generation error: {str(e)}"
            )
        finally:
            if not self.llm.callbacks[0].done.is_set():
                self.llm.callbacks[0].done.set()

    async def process_chat_request(
        self, 
        content: str, 
        websearch: bool = False, 
        reasoning: bool = False
    ) -> StreamingResponse:
        """Process a chat request and return a token-by-token streaming response."""
        try:
            self.websearch = websearch
            self.reasoning = reasoning
            return StreamingResponse(
                self.generate_response(content),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        except Exception as e:
            logger.error(f"Request processing error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Request processing error: {str(e)}"
            )
