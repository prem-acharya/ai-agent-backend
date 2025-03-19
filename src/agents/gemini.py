import os
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
            await self.reset_callback()
            
            # Determine request type based on user intent
            content_lower = content.lower()
            
            # Check for event/meeting keywords
            event_keywords = ["meeting", "schedule meeting", "create meeting", "set meeting", 
                            "event", "create event", "set event", "calendar event"]
            is_event = any(keyword in content_lower for keyword in event_keywords)
            
            # Check for task/reminder keywords
            task_keywords = ["reminder", "remind me", "create task", "set task", "set reminder", 
                           "create reminder", "todo", "task"]
            is_task = any(keyword in content_lower for keyword in task_keywords)
            
            # Handle both task and event
            if is_task and is_event:
                logger.info("Processing both task and event request")
                
                # First create the task
                task_data = self._prepare_task_data(content)
                if task_data:
                    for tool in self.tools:
                        if tool.name == "create_task":
                            result_json = await tool._arun(json.dumps(task_data))
                            try:
                                result_data = json.loads(result_json)
                                if result_data.get("success"):
                                    yield f"✅ Task created successfully!\n\nTask Details:\n```json\n{json.dumps(task_data, indent=2)}\n```\n\n"
                                else:
                                    yield f"❌ Failed to create task: {result_data.get('error')}\n\n"
                            except Exception as e:
                                yield f"❌ Error processing task creation: {str(e)}\n\n"
                
                # Then create the event
                event_data = self._prepare_event_data(content)
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
            
            # Handle event/meeting only
            elif is_event:
                logger.info("Processing event/meeting request")
                event_data = self._prepare_event_data(content)
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
            
            # Handle task/reminder only
            elif is_task:
                logger.info("Processing task/reminder request")
                task_data = self._prepare_task_data(content)
                if task_data:
                    for tool in self.tools:
                        if tool.name == "create_task":
                            result_json = await tool._arun(json.dumps(task_data))
                            try:
                                result_data = json.loads(result_json)
                                if result_data.get("success"):
                                    yield f"✅ Task/Reminder created successfully!\n\nTask Details:\n```json\n{json.dumps(task_data, indent=2)}\n```"
                                else:
                                    yield f"❌ Failed to create task: {result_data.get('error')}"
                            except Exception as e:
                                yield f"❌ Error processing task creation: {str(e)}"
            
            # Handle other queries
            else:
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

    def _prepare_task_data(self, content: str) -> dict:
        """Prepare task data from user input"""
        task_data = {}
        content_lower = content.lower()
        
        # Extract task title
        task_title = None
        task_markers = ["remind me to ", "set reminder to ", "create task to ", "set task to "]
        for marker in task_markers:
            if marker in content_lower:
                task_title = content_lower.split(marker, 1)[1].strip()
                break
        
        if not task_title:
            # Fallback extraction
            if "to " in content_lower:
                task_title = content_lower.split("to ", 1)[1].strip()
            else:
                task_title = content_lower.replace("reminder", "").replace("task", "").replace("set", "").replace("create", "").strip()
        
        task_data["title"] = task_title.title()
        
        # Extract due date
        if any(day in content_lower for day in ["tomorrow", "tmr"]):
            task_data["due"] = "tomorrow"
        elif "next week" in content_lower:
            from datetime import datetime, timedelta
            task_data["due"] = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        elif "today" in content_lower:
            task_data["due"] = "today"
        else:
            # Try to find a specific date
            import re
            from datetime import datetime
            
            date_patterns = [
                (r'(\d{2})/(\d{2})(?:/\d{4})?', '%d/%m/%Y'),  # DD/MM or DD/MM/YYYY
                (r'(\d{2})-(\d{2})(?:-\d{4})?', '%d-%m-%Y'),  # DD-MM or DD-MM-YYYY
                (r'\d{4}-\d{2}-\d{2}', '%Y-%m-%d'),           # YYYY-MM-DD
                (r'\d{2}/\d{2}/\d{4}', '%d/%m/%Y')            # DD/MM/YYYY
            ]
            
            for pattern, date_format in date_patterns:
                if matches := re.search(pattern, content):
                    try:
                        if len(matches.groups()) == 2:  # DD/MM format without year
                            day, month = matches.groups()
                            current_year = datetime.now().year
                            date_str = f"{day}/{month}/{current_year}"
                            parsed_date = datetime.strptime(date_str, '%d/%m/%Y')
                        else:
                            date_str = matches.group(0)
                            if len(date_str.split('/')[0]) == 4:  # YYYY/MM/DD format
                                parsed_date = datetime.strptime(date_str, '%Y/%m/%d')
                            else:
                                parsed_date = datetime.strptime(date_str, date_format)
                        
                        task_data["due"] = parsed_date.strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        continue
        
        # Add notes if available
        if "notes:" in content_lower:
            notes = content.split("notes:", 1)[1].strip()
            task_data["notes"] = notes
        
        return task_data

    def _prepare_event_data(self, content: str) -> dict:
        """Prepare event data from user input"""
        event_data = {}
        content_lower = content.lower()
        
        # Extract event title
        event_title = None
        event_markers = ["schedule meeting for ", "create meeting for ", "set meeting for ",
                        "schedule event for ", "create event for ", "set event for "]
        for marker in event_markers:
            if marker in content_lower:
                event_title = content_lower.split(marker, 1)[1].strip()
                break
        
        if not event_title:
            # Fallback extraction
            if "meeting about " in content_lower:
                event_title = content_lower.split("meeting about ", 1)[1].strip()
            elif "event about " in content_lower:
                event_title = content_lower.split("event about ", 1)[1].strip()
            else:
                event_title = content_lower.replace("meeting", "").replace("event", "").replace("schedule", "").replace("create", "").replace("set", "").strip()
        
        event_data["title"] = event_title.title()
        
        # Extract date
        if any(day in content_lower for day in ["tomorrow", "tmr"]):
            event_data["due"] = "tomorrow"
        elif "next week" in content_lower:
            from datetime import datetime, timedelta
            event_data["due"] = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        elif "today" in content_lower:
            event_data["due"] = "today"
        else:
            # Try to find a specific date
            import re
            from datetime import datetime
            
            date_patterns = [
                (r'(\d{2})/(\d{2})(?:/\d{4})?', '%d/%m/%Y'),  # DD/MM or DD/MM/YYYY
                (r'(\d{2})-(\d{2})(?:-\d{4})?', '%d-%m-%Y'),  # DD-MM or DD-MM-YYYY
                (r'\d{4}-\d{2}-\d{2}', '%Y-%m-%d'),           # YYYY-MM-DD
                (r'\d{2}/\d{2}/\d{4}', '%d/%m/%Y')            # DD/MM/YYYY
            ]
            
            for pattern, date_format in date_patterns:
                if matches := re.search(pattern, content):
                    try:
                        if len(matches.groups()) == 2:  # DD/MM format without year
                            day, month = matches.groups()
                            current_year = datetime.now().year
                            date_str = f"{day}/{month}/{current_year}"
                            parsed_date = datetime.strptime(date_str, '%d/%m/%Y')
                        else:
                            date_str = matches.group(0)
                            if len(date_str.split('/')[0]) == 4:  # YYYY/MM/DD format
                                parsed_date = datetime.strptime(date_str, '%Y/%m/%d')
                            else:
                                parsed_date = datetime.strptime(date_str, date_format)
                        
                        event_data["due"] = parsed_date.strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        continue
        
        # Extract time range
        time_range_pattern = r'(\d{1,2})(?::\d{2})?\s*(?:am|pm)\s*to\s*(\d{1,2})(?::\d{2})?\s*(?:am|pm)'
        if time_match := re.search(time_range_pattern, content_lower):
            start_hour, end_hour = time_match.groups()
            
            # Convert to 24-hour format
            if "pm" in content_lower.split("to")[0]:
                start_hour = str(int(start_hour) + 12) if int(start_hour) < 12 else start_hour
            if "pm" in content_lower.split("to")[1]:
                end_hour = str(int(end_hour) + 12) if int(end_hour) < 12 else end_hour
            
            event_data["start_time"] = f"{int(start_hour):02d}:00"
            event_data["end_time"] = f"{int(end_hour):02d}:00"
        
        # Extract attendees
        if "guest list:" in content_lower:
            attendees = content.lower().split("guest list:")[1].strip().split(",")
            event_data["attendees"] = [email.strip() for email in attendees]
        elif "email is" in content_lower:
            email = content.lower().split("email is")[1].strip()
            event_data["attendees"] = [email]
        
        # Add notes if available
        if "notes:" in content_lower:
            notes = content.split("notes:", 1)[1].strip()
            event_data["notes"] = notes
        
        return event_data

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
