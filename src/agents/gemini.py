import os
import re
import asyncio
import random
from typing import AsyncIterable, Optional, List, Dict, Any
import logging
import json
import uuid
from datetime import datetime, timedelta

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
from src.utils.task_utils import prepare_task_data, format_task_details
from src.utils.event_utils import prepare_event_data, format_event_details
from src.utils.time_utils import parse_date_from_text, parse_time_range, format_task_date
from src.utils.prompt.task_prompts import get_task_analysis_prompt
from src.utils.prompt.event_prompts import get_event_analysis_prompt

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

    async def _get_tasks(self, content: str) -> Dict[str, Any]:
        """Get tasks based on user request."""
        try:
            content_lower = content.lower()
            
            # Determine time period
            today_keywords = ["today", "today's", "for today"]
            tomorrow_keywords = ["tomorrow", "tomorrow's", "for tomorrow"]
            
            today_only = any(keyword in content_lower for keyword in today_keywords)
            tomorrow_only = any(keyword in content_lower for keyword in tomorrow_keywords)
            
            # Prepare query for GetTasksTool
            query = {}
            
            if today_only:
                query["today_only"] = True
            elif tomorrow_only:
                query["tomorrow_only"] = True
                logger.info("Request for tomorrow's tasks detected")
            else:
                # Default: get all tasks
                query["today_only"] = False
                query["tomorrow_only"] = False
            
            # Find and use the GetTasksTool
            for tool in self.tools:
                if tool.name == "get_tasks":
                    result_json = await tool._arun(json.dumps(query))
                    return json.loads(result_json)
            
            return {"success": False, "error": "Task retrieval tool not available"}
        except Exception as e:
            logger.error(f"Error getting tasks: {str(e)}")
            return {"success": False, "error": str(e)}

    async def generate_response(self, content: str) -> AsyncIterable[str]:
        """Generate a response using the agent."""
        try:
            await self.reset_callback()
            
            # Determine request type based on user intent
            content_lower = content.lower()
            
            # Check for task retrieval request
            get_task_keywords = ["show tasks", "list out my tasks", "get tasks", "list tasks", "view tasks", "what are my tasks", 
                               "show my tasks", "display tasks", "check tasks", "list my tasks", "show my tasks", "check my tasks"]
            
            # Add keywords for specific time periods
            time_specific_task_keywords = [
                "tomorrow's tasks", "tomorrow tasks", "tasks for tomorrow", 
                "today's tasks", "today tasks", "tasks for today",
                "upcoming tasks", "next tasks", "show me my"
            ]
            
            # Check if the request is about viewing tasks
            is_get_tasks = any(keyword in content_lower for keyword in get_task_keywords)
            is_time_specific_tasks = any(keyword in content_lower for keyword in time_specific_task_keywords)
            
            if is_get_tasks or is_time_specific_tasks:
                logger.info("Processing task retrieval request")
                tasks_result = await self._get_tasks(content)
                
                if tasks_result.get("success"):
                    yield "📋 Here are your tasks:\n\n"
                    
                    if not tasks_result.get("tasks"):
                        yield "No tasks found for your request. You can create new tasks by saying something like 'create a task to review the project'.\n\n"
                        return
                    
                    yield f"Task List: {tasks_result.get('task_list', 'Default')}\n\n"
                    
                    for task in tasks_result.get("tasks", []):
                        status_emoji = "✅" if task["status"] == "completed" else "⏳"
                        task_line = f"{status_emoji} {task['title']}"
                        
                        if task.get("due"):
                            formatted_date = format_task_date(task["due"])
                            task_line += f"\n   📅 Due: {formatted_date}"
                        
                        if task.get("notes"):
                            task_line += f"\n   📝 Notes: {task['notes']}"
                        
                        yield f"{task_line}\n\n"
                else:
                    yield f"❌ Failed to retrieve tasks: {tasks_result.get('error', 'Unknown error')}"
                return

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
                            "event", "create event", "set event", "calendar event",
                            "appointment", "schedule", "interview", "call", "conference",
                            "webinar", "session", "catch up", "sync", "discussion"]
            is_event = any(keyword in content_lower for keyword in event_keywords)
            
            # Handle event/meeting request
            if is_event:
                logger.info("Processing event/meeting request")
                event_data = await prepare_event_data(content, self.llm)
                logger.info(f"Prepared event data: {json.dumps(event_data, indent=2)}")
                
                if event_data:
                    for tool in self.tools:
                        if tool.name == "create_event":
                            # First yield the AI-generated event details
                            yield "🤖 Here's how I understand your event:\n\n"
                            yield format_event_details(event_data)
                            yield "\n\n⏳ Creating the event..."
                            
                            # Create the event
                            result_json = await tool._arun(json.dumps(event_data))
                            try:
                                result_data = json.loads(result_json)
                                if result_data.get("success"):
                                    yield "\n\n✅ Event created successfully!"
                                    
                                    # Add Google Meet link if available
                                    if (result_data.get("event") and 
                                        result_data["event"].get("hangout_link")):
                                        yield f"\n\n🔗 **Google Meet Link:** {result_data['event']['hangout_link']}"
                                        
                                    # Add calendar link
                                    if (result_data.get("event") and 
                                        result_data["event"].get("calendar_link")):
                                        yield f"\n\n📆 **Calendar Link:** {result_data['event']['calendar_link']}"
                                else:
                                    yield f"\n\n❌ Failed to create event: {result_data.get('error')}"
                            except Exception as e:
                                yield f"\n\n❌ Error processing event creation: {str(e)}"
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
