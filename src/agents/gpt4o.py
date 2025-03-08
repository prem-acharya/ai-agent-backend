from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncIterable
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain
import asyncio
from src.utils.gpt4o_streaming import BaseStreamingLLM
from src.tools.datetime.time_tool import CurrentTimeTool
from src.tools.websearch.websearch_tool import WebSearchTool
from src.utils.prompts import initialize_prompts
from functools import lru_cache
from langchain.callbacks import AsyncIteratorCallbackHandler

class GPT4OAgent(BaseStreamingLLM):
    """GPT-4 Optimized Agent with streaming capabilities"""

    def __init__(self, websearch: bool = False, reasoning: bool = False):
        super().__init__()
        self.websearch = websearch
        self.reasoning = reasoning
        self.cot_prompt, self.direct_prompt, self.final_prompt = initialize_prompts()
        self._initialize_chains()
        if websearch:
            self.tools = [CurrentTimeTool(), WebSearchTool()]
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=True
            )

    @lru_cache(maxsize=2)
    def _initialize_chains(self):
        """Initialize LLM chains with separate callbacks"""
        self.cot_chain = LLMChain(llm=self.llm, prompt=self.cot_prompt)
        self.direct_chain = LLMChain(llm=self.llm, prompt=self.direct_prompt)
        self.final_chain = LLMChain(llm=self.llm, prompt=self.final_prompt)

    async def _reset_callback(self):
        """Reset callback handler"""
        self.callback.done.set()
        await asyncio.sleep(0.1)  # Small delay to ensure completion
        self.callback.done.clear()
        self.callback = AsyncIteratorCallbackHandler()
        self.llm.callbacks = [self.callback]

    async def generate_response(self, content: str) -> AsyncIterable[str]:
        """Generate streaming response from the model"""
        try:
            if self.websearch:
                web_tool = WebSearchTool()
                web_results = await web_tool._arun(content)
                content = f"{content}\nContext from search: {web_results}"

            if self.reasoning:
                # Chain of thought reasoning
                yield "🤔 Reasoning Process:\n\n"
                
                # Get reasoning
                self.callback = AsyncIteratorCallbackHandler()
                self.llm.callbacks = [self.callback]
                reasoning_task = asyncio.create_task(
                    self.cot_chain.arun(question=content)
                )
                
                async for token in self.stream_tokens():
                    yield token
                
                reasoning_result = await reasoning_task
                await self._reset_callback()
                
                # Final answer
                yield "\n\n✨ Final Answer:\n\n"
                
                # Stream final answer
                final_task = asyncio.create_task(
                    self.final_chain.arun(chain_of_thought=reasoning_result)
                )
                
                async for token in self.stream_tokens():
                    yield token
                
                await final_task
                
            else:
                # Direct response
                direct_task = asyncio.create_task(
                    self.direct_chain.arun(question=content)
                )
                
                async for token in self.stream_tokens():
                    yield token
                
                await direct_task

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Generation error: {str(e)}"
            )
        finally:
            self.callback.done.set()

    async def process_chat_request(self, content: str, websearch: bool = False, reasoning: bool = False) -> StreamingResponse:
        """Process chat request and return streaming response"""
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
            raise HTTPException(
                status_code=500,
                detail=f"Request processing error: {str(e)}"
            )