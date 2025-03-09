from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncIterable
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import initialize_agent, AgentType
from langchain_core.runnables import RunnablePassthrough
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
        """Initialize runnable sequences"""
        # Create runnable sequences using the pipe operator
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
        """Reset callback handler"""
        self.callback.done.set()
        await asyncio.sleep(0.1)
        self.callback.done.clear()
        self.callback = AsyncIteratorCallbackHandler()
        self.llm.callbacks = [self.callback]

    async def generate_response(self, content: str) -> AsyncIterable[str]:
        """Generate streaming response from the model"""
        try:
            if self.websearch:
                # First get current time
                time_tool = CurrentTimeTool()
                current_time = await time_tool._arun()

                # Then perform web search
                yield "🌐 Searching the web...\n\n"
                web_tool = WebSearchTool()
                web_results = await web_tool._arun(content)
                
                # Combine context
                content = f"""
Current Time: {current_time}
User Question: {content}
Web Search Results: {web_results}
"""

            if self.reasoning:
                yield "reasoning:\n\n"
                
                # Get reasoning with enhanced context
                self.callback = AsyncIteratorCallbackHandler()
                self.llm.callbacks = [self.callback]
                reasoning_task = asyncio.create_task(
                    self.cot_chain.ainvoke({"question": content})
                )
                
                async for token in self.stream_tokens():
                    yield token
                
                reasoning_result = await reasoning_task
                reasoning_text = reasoning_result['text']
                await self._reset_callback()
                
                # Final answer incorporating all context
                yield "\n\nFinal Answer based on analysis:\n\n"
                
                final_task = asyncio.create_task(
                    self.final_chain.ainvoke({
                        "chain_of_thought": reasoning_text,
                        "web_context": web_results if self.websearch else ""
                    })
                )
                
                async for token in self.stream_tokens():
                    yield token
                
                await final_task
                
            else:
                # Direct response
                direct_task = asyncio.create_task(
                    self.direct_chain.ainvoke({"question": content})
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