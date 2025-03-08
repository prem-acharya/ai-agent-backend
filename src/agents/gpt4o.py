from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncIterable
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import initialize_agent, AgentType
import asyncio
from src.utils.gpt4o_streaming import BaseStreamingLLM
from src.tools.datetime.time_tool import CurrentTimeTool
from src.tools.websearch.websearch_tool import WebSearchTool

class GPT4OAgent(BaseStreamingLLM):
    """GPT-4 Optimized Agent with streaming capabilities"""

    def __init__(self, websearch: bool = False):
        super().__init__()
        self.websearch = websearch
        if websearch:
            self.tools = [CurrentTimeTool(), WebSearchTool()]
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=True
            )

    async def generate_response(self, content: str) -> AsyncIterable[str]:
        """Generate streaming response from the model"""
        try:
            if self.websearch:
                # First get current time
                time_tool = CurrentTimeTool()
                current_time = await time_tool._arun()
                
                # Create system message with current time context
                system_msg = SystemMessage(content=f"Current context: {current_time}")
                user_msg = HumanMessage(content=content)
                messages = [system_msg, user_msg]
            else:
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

    async def process_chat_request(self, content: str, websearch: bool = False) -> StreamingResponse:
        """Process chat request and return streaming response"""
        try:
            self.websearch = websearch
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