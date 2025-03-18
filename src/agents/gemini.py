import os
import asyncio
from typing import AsyncIterable, Optional, List, Dict, Any
import logging
import json

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

from src.tools.google.create_task_tool import CreateTaskTool
from src.tools.google.get_tasks_tool import GetTasksTool
from src.utils.gemini_streaming import BaseGeminiStreaming

logger = logging.getLogger(__name__)

class GeminiAgent(BaseGeminiStreaming):
    """Gemini Agent with task management capabilities."""

    def __init__(self, google_access_token: Optional[str] = None):
        super().__init__()
        self.google_access_token = google_access_token
        
        # Initialize tools
        self.tools = []
        
        # Add task tools if access token is provided
        if google_access_token:
            self.tools.extend([
                CreateTaskTool(google_access_token),
                GetTasksTool(google_access_token)
            ])
            logger.info("Task tools initialized successfully")
        
        # Initialize agent if tools are available
        if self.tools:
            logger.info(f"Initializing agent with {len(self.tools)} tools")
            
            # Create a custom prompt for the agent
            agent_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant that manages tasks. When users interact with you:

1. For questions about tasks:
   - Use the get_tasks tool
   - Example: {"today_only": true} to get today's tasks
   - Format the response nicely with emojis and clear status

2. For creating tasks:
   - Use the create_task tool
   - Required: title
   - Optional: due date (today/tomorrow/YYYY-MM-DD), notes
   - Example: {"title": "Buy groceries", "due": "tomorrow", "notes": "Get milk"}
   - Confirm task creation with a friendly message

Always use the appropriate tool for task operations. Don't pretend to create or list tasks."""),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                agent_kwargs={"prompt": agent_prompt},
                verbose=True
            )
            logger.info("Agent initialization complete")
        else:
            logger.warning("No tools available for agent")
    
    def _extract_task_data_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Extract task creation data from response text."""
        tasks = []
        try:
            import re
            json_patterns = [
                r'```(?:json)?\s*({[^}]+})\s*```',  # JSON in code blocks
                r'({[\s\S]*?"action"\s*:\s*"create_task"[\s\S]*?})'  # Direct JSON with create_task action
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response)
                for match in matches:
                    try:
                        task_data = json.loads(match)
                        if task_data.get("action") == "create_task":
                            tasks.append(task_data)
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Found {len(tasks)} task creation objects in response")
            return tasks
        except Exception as e:
            logger.error(f"Error extracting task data: {str(e)}")
            return []

    async def generate_response(self, content: str) -> AsyncIterable[str]:
        """Generate a response using the agent."""
        try:
            await self._reset_callback()
            collected_response = ""
            
            # Check if this is a task-related request
            if self.google_access_token and any(word in content.lower() for word in ["task", "todo", "reminder"]):
                logger.info(f"Processing task-related request: {content}")
                try:
                    if self.agent:
                        agent_response = await self.agent.arun(content)
                        collected_response = agent_response
                        
                        # Try to extract task data from the response
                        task_objects = self._extract_task_data_from_response(agent_response)
                        
                        if task_objects:
                            results = []
                            for task_data in task_objects:
                                try:
                                    logger.info(f"Creating task: {task_data.get('title')}")
                                    for tool in self.tools:
                                        if tool.name == "create_task":
                                            result_json = await tool._arun(json.dumps(task_data))
                                            try:
                                                result_data = json.loads(result_json)
                                                if result_data.get("success"):
                                                    task_title = task_data.get("title", "Unknown")
                                                    results.append(f"✅ I've created the task \"{task_title}\" for you in Google Tasks.")
                                            except Exception:
                                                results.append(result_json)
                                except Exception as e:
                                    logger.error(f"Error creating task: {str(e)}")
                            
                            # Yield original response plus task creation confirmations
                            if results:
                                yield f"{collected_response}\n\n{results[0]}"  # Only return the first task confirmation
                                return
                            
                        # If no tasks were created or there was an error
                        yield collected_response
                        
                    else:
                        yield "⚠️ Task management is not available at the moment."
                except Exception as e:
                    logger.exception(f"Error handling task request: {str(e)}")
                    yield f"⚠️ I encountered an error while processing your task request: {str(e)}"
            else:
                # For non-task queries, use direct response
                response = await self.llm.agenerate([content])
                yield response.generations[0][0].text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if not self.llm.callbacks[0].done.is_set():
                self.llm.callbacks[0].done.set()

    async def process_chat_request(self, content: str) -> StreamingResponse:
        """Process a chat request and return a streaming response."""
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
            logger.error(f"Request processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
