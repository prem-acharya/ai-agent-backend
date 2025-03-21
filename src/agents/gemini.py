import os
import re
import asyncio
import random
from typing import AsyncIterable, Optional, List, Dict, Any
import logging
import json

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

from src.tools.google.create_task_tool import CreateTaskTool
from src.tools.google.create_event_tool import CreateEventTool
from src.tools.google.get_tasks_tool import GetTasksTool
from src.utils.gemini_streaming import BaseGeminiStreaming
from src.utils.prompts import initialize_prompts
from src.utils.task_utils import prepare_task_data, prepare_event_data, format_task_details
from src.utils.time_utils import parse_date_from_text
from src.utils.task.task_prompts import get_task_analysis_prompt

logger = logging.getLogger(__name__)

class GeminiAgent(BaseGeminiStreaming):
    """Gemini Agent with task management capabilities."""

    def __init__(self, websearch: bool = False, reasoning: bool = False, google_access_token: Optional[str] = None):
        super().__init__()
        self.websearch = websearch
        self.reasoning = reasoning
        self.google_access_token = google_access_token
        
        # Initialize tools
        self.tools = []
        
        # Add task and event tools if access token is provided
        if google_access_token:
            self.tools.extend([
                CreateTaskTool(google_access_token),
                CreateEventTool(google_access_token),
                GetTasksTool(google_access_token)
            ])
            logger.info("Task and event tools initialized successfully")
        
        # Initialize agent if tools are available
        if self.tools:
            logger.info(f"Initializing agent with {len(self.tools)} tools")
            
            # Get task management prompt
            _, _, _, task_management_prompt = initialize_prompts()
            
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                agent_kwargs={"prompt": task_management_prompt},
                verbose=True
            )
            logger.info("Agent initialization complete")
        else:
            logger.warning("No tools available for agent")
    
    def _extract_task_data_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Extract task creation data from response text."""
        tasks = []
        try:
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

    async def _prepare_task_data(self, content: str) -> Dict[str, Any]:
        """Prepare task data from user input using AI analysis."""
        try:
            # Get AI analysis of the task
            analysis_prompt = get_task_analysis_prompt().format(content=content)
            response = await self.llm.agenerate([[HumanMessage(content=analysis_prompt)]])
            response_text = response.generations[0][0].text.strip()
            
            # Try to extract JSON from the response
            try:
                # Clean up the response text to ensure valid JSON
                json_text = re.search(r'\{[\s\S]*\}', response_text)
                if json_text:
                    task_analysis = json.loads(json_text.group(0))
                else:
                    logger.error("No JSON found in response")
                    return prepare_task_data(content)
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                return prepare_task_data(content)
            
            # Convert task analysis to task data format
            task_data = {
                "title": task_analysis.get("title", "New Task"),
                "due": parse_date_from_text(content),
                "notes": task_analysis.get("notes", "")
            }
            
            # Handle notes based on type (string or list)
            if isinstance(task_data["notes"], list):
                # If it's a list, join with proper newlines
                task_data["notes"] = "\n".join([note.strip() for note in task_data["notes"] if note.strip()])
            
            # Add description to notes if available
            if task_analysis.get("description"):
                if task_data["notes"]:
                    task_data["notes"] = f"{task_analysis['description']}\n\n{task_data['notes']}"
                else:
                    task_data["notes"] = task_analysis['description']
            
            return task_data
            
        except Exception as e:
            logger.error(f"Error getting task analysis from Gemini: {str(e)}")
            return prepare_task_data(content)

    async def generate_response(self, content: str) -> AsyncIterable[str]:
        """Generate a response using the agent."""
        try:
            await self.reset_callback()
            
            # Determine request type based on user intent
            content_lower = content.lower()
            
            # Check for task/reminder keywords first
            task_keywords = ["reminder", "remind me", "create task", "set task", "set reminder", 
                           "create reminder", "todo", "task"]
            is_task = any(keyword in content_lower for keyword in task_keywords)
            
            # If it's a task request, handle it directly without checking for events
            if is_task:
                logger.info("Processing task/reminder request")
                task_data = await self._prepare_task_data(content)
                logger.info(f"Prepared task data: {json.dumps(task_data, indent=2)}")
                
                if task_data:
                    for tool in self.tools:
                        if tool.name == "create_task":
                            # First yield the AI-generated task details
                            yield "🤖 Here's how I understand your task:\n\n"
                            yield format_task_details(task_data)
                            yield "\n\n⏳ Creating the task..."
                            
                            # Create the task
                            result_json = await tool._arun(json.dumps(task_data))
                            try:
                                result_data = json.loads(result_json)
                                if result_data.get("success"):
                                    yield "\n\n✅ Task created successfully!"
                                else:
                                    yield f"\n\n❌ Failed to create task: {result_data.get('error')}"
                            except Exception as e:
                                yield f"\n\n❌ Error processing task creation: {str(e)}"
                return
            
            # Check for event/meeting keywords
            event_keywords = ["meeting", "schedule meeting", "create meeting", "set meeting", 
                            "event", "create event", "set event", "calendar event"]
            is_event = any(keyword in content_lower for keyword in event_keywords)
            
            # Handle event/meeting request
            if is_event:
                logger.info("Processing event/meeting request")
                event_data = prepare_event_data(content)
                if event_data:
                    for tool in self.tools:
                        if tool.name == "create_event":
                            result_json = await tool._arun(json.dumps(event_data))
                            try:
                                result_data = json.loads(result_json)
                                if result_data.get("success"):
                                    yield f"📅 Meeting/Event created successfully!\n\nEvent Details:\n```json\n{json.dumps(event_data, indent=2)}\n```"
                                else:
                                    yield f"❌ Failed to create event: {result_data.get('error')}"
                            except Exception as e:
                                yield f"❌ Error processing event creation: {str(e)}"
                return
            
            # Handle other queries
            if self.reasoning:
                yield "reasoning start\n\n"
                response = await self.llm.agenerate([[HumanMessage(content=f"Think step by step to answer this question: {content}")]])
                reasoning_text = response.generations[0][0].text
                yield reasoning_text
                
                yield "\n\nFinal Answer start\n\n"
                summary_prompt = f"Based on the above reasoning, provide a concise final answer to the original question: {content}"
                response = await self.llm.agenerate([[HumanMessage(content=summary_prompt)]])
                yield response.generations[0][0].text
            else:
                response = await self.llm.agenerate([[HumanMessage(content=content)]])
                yield response.generations[0][0].text
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if hasattr(self.callback, "done") and not self.callback.done.is_set():
                self.callback.done.set()

    async def process_chat_request(self, content: str, websearch: bool = False, reasoning: bool = False) -> StreamingResponse:
        """Process a chat request and return a streaming response."""
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
            logger.error(f"Request processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
