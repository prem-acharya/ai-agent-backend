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
from src.tools.google.get_events_tool import GetEventsTool
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
                GetTasksTool(google_access_token),
                GetEventsTool(google_access_token)
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

    async def _get_events(self, content: str) -> Dict[str, Any]:
        """Get events based on user request."""
        try:
            content_lower = content.lower()
            
            # Determine time period
            today_keywords = ["today", "today's", "for today"]
            tomorrow_keywords = ["tomorrow", "tomorrow's", "for tomorrow"]
            upcoming_keywords = ["upcoming", "next", "coming", "future"]
            
            today_only = any(keyword in content_lower for keyword in today_keywords)
            tomorrow_only = any(keyword in content_lower for keyword in tomorrow_keywords)
            upcoming_only = any(keyword in content_lower for keyword in upcoming_keywords)
            
            # Prepare query for GetEventsTool
            query = {}
            
            if today_only:
                query["today_only"] = True
            elif tomorrow_only:
                query["tomorrow_only"] = True
                logger.info("Request for tomorrow's events detected")
            elif upcoming_only:
                query["upcoming_only"] = True
                logger.info("Request for upcoming events detected")
            else:
                # Default: get upcoming events
                query["upcoming_only"] = True
                query["max_results"] = 10
            
            # Find and use the GetEventsTool
            for tool in self.tools:
                if tool.name == "get_events":
                    result_json = await tool._arun(json.dumps(query))
                    return json.loads(result_json)
            
            return {"success": False, "error": "Event retrieval tool not available"}
        except Exception as e:
            logger.error(f"Error getting events: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _get_tasks_and_events(self, content: str) -> Dict[str, Any]:
        """Get both tasks and events based on user request."""
        try:
            # Get tasks and events separately
            tasks_result = await self._get_tasks(content)
            events_result = await self._get_events(content)
            
            # Combine results
            combined_result = {
                "success": tasks_result.get("success") or events_result.get("success"),
                "tasks": tasks_result.get("tasks", []) if tasks_result.get("success") else [],
                "events": events_result.get("events", []) if events_result.get("success") else [],
                "task_list": tasks_result.get("task_list", "Default") if tasks_result.get("success") else None,
            }
            
            # Add errors if any
            if not tasks_result.get("success"):
                combined_result["tasks_error"] = tasks_result.get("error")
            if not events_result.get("success"):
                combined_result["events_error"] = events_result.get("error")
                
            return combined_result
        except Exception as e:
            logger.error(f"Error getting tasks and events: {str(e)}")
            return {"success": False, "error": str(e)}

    async def generate_response(self, content: str) -> AsyncIterable[str]:
        """Generate a response using the agent."""
        try:
            await self.reset_callback()
            
            # Determine request type based on user intent
            content_lower = content.lower()
            logger.info(f"Processing user request: '{content_lower}'")
            
            # Initialize intent variables
            is_get_tasks = False
            is_time_specific_tasks = False
            is_get_events = False
            is_time_specific_events = False
            is_get_both = False
            is_task = False
            is_event = False
            
            # Direct pattern matching for common retrieval phrases
            if any(pattern in content_lower for pattern in [
                "show my", "show all", "list my", "show me", "get my", "what are my", 
                "all my", "my all", "all the", "see my", "view my", "check my"
            ]):
                logger.info("Detected direct retrieval phrase")
                if "meeting" in content_lower or "meetings" in content_lower or "event" in content_lower or "events" in content_lower:
                    logger.info("Direct retrieval phrase for events/meetings")
                    is_get_events = True
                if "task" in content_lower or "tasks" in content_lower or "todo" in content_lower or "to-do" in content_lower or "to do" in content_lower:
                    logger.info("Direct retrieval phrase for tasks")
                    is_get_tasks = True
                # If both are mentioned, prioritize the combined view
                if ("meeting" in content_lower or "meetings" in content_lower or "event" in content_lower or "events" in content_lower) and \
                   ("task" in content_lower or "tasks" in content_lower or "todo" in content_lower or "to-do" in content_lower or "to do" in content_lower):
                    logger.info("Direct retrieval phrase for both tasks and events")
                    is_get_both = True
                    # Make sure we don't trigger the individual retrievals
                    is_get_events = False
                    is_get_tasks = False
            
            # Check for task retrieval request
            get_task_keywords = ["show tasks", "list out my tasks", "get tasks", "list tasks", "view tasks", "what are my tasks", 
                               "show my tasks", "display tasks", "check tasks", "list my tasks", "show my tasks", "check my tasks"]
            
            # Add keywords for specific time periods for tasks
            time_specific_task_keywords = [
                "tomorrow's tasks", "tomorrow tasks", "tasks for tomorrow", 
                "today's tasks", "today tasks", "tasks for today",
                "upcoming tasks", "next tasks", "show me my"
            ]

            # Check for event retrieval request
            get_event_keywords = ["show events", "list events", "get events", "view events", "what are my events",
                                "show my events", "display events", "check events", "list my events", "show meetings",
                                "list meetings", "show my meetings", "check my calendar", "view calendar", "calendar events",
                                "show all meetings", "show my all meetings", "show all my meetings", "show my all the meetings",
                                "all meetings", "all events", "all my meetings", "all my events", "all of my meetings", 
                                "all of my events", "meetings", "my meetings"]
            
            # Add keywords for specific time periods for events
            time_specific_event_keywords = [
                "tomorrow's events", "tomorrow events", "events for tomorrow",
                "today's events", "today events", "events for today",
                "upcoming events", "next events", "show me my events",
                "tomorrow's meetings", "today's meetings", "upcoming meetings"
            ]
            
            # Check for combined task and event requests
            get_both_keywords = ["show my schedule", "show my today schedule", "show my tomorrow schedule", "what's on my schedule", "check my schedule",
                               "what do i have", "view my schedule", "list everything", "show everything",
                               "all tasks and events", "all events and tasks", "my agenda", "what's on my agenda"]
            
            # Check if the request is about viewing tasks
            is_get_tasks = any(keyword in content_lower for keyword in get_task_keywords)
            is_time_specific_tasks = any(keyword in content_lower for keyword in time_specific_task_keywords)
            
            # Check if the request is about viewing events
            is_get_events = any(keyword in content_lower for keyword in get_event_keywords)
            is_time_specific_events = any(keyword in content_lower for keyword in time_specific_event_keywords)
            
            # Check if the request is about viewing both tasks and events
            is_get_both = any(keyword in content_lower for keyword in get_both_keywords)
            
            # Log detection results for debugging
            logger.info(f"Intent detection: get_tasks={is_get_tasks}, time_specific_tasks={is_time_specific_tasks}")
            logger.info(f"Intent detection: get_events={is_get_events}, time_specific_events={is_time_specific_events}")
            logger.info(f"Intent detection: get_both={is_get_both}")
            
            # Special handling for ambiguous request like "meetings" without clear context
            if content_lower in ["meetings", "events", "tasks"]:
                logger.info("Processing ambiguous retrieval request as view request")
                is_get_events = True if content_lower in ["meetings", "events"] else is_get_events
                is_get_tasks = True if content_lower == "tasks" else is_get_tasks
            
            # Special handling for "all" phrases
            if "all" in content_lower:
                if "meeting" in content_lower or "meetings" in content_lower or "event" in content_lower or "events" in content_lower:
                    logger.info("Processing 'all meetings/events' as a retrieval request")
                    is_get_events = True
                if "task" in content_lower or "tasks" in content_lower:
                    logger.info("Processing 'all tasks' as a retrieval request")
                    is_get_tasks = True
                if ("meetings" in content_lower or "events" in content_lower) and ("task" in content_lower or "tasks" in content_lower):
                    logger.info("Processing 'all tasks and events' as a combined retrieval request")
                    is_get_both = True
            
            # Handle requests for both tasks and events
            if is_get_both:
                logger.info("Processing combined task and event retrieval request")
                combined_result = await self._get_tasks_and_events(content)
                
                if combined_result.get("success"):
                    yield "📋 Here's your schedule:\n\n"
                    
                    # Check if we have both tasks and events
                    has_tasks = bool(combined_result.get("tasks"))
                    has_events = bool(combined_result.get("events"))
                    
                    if not has_tasks and not has_events:
                        yield "No upcoming tasks or events found for your request.\n\n"
                        return
                    
                    # Display tasks first if we have any
                    if has_tasks:
                        yield "🗒️ **TASKS**\n\n"
                        yield f"Task List: {combined_result.get('task_list', 'Default')}\n\n"
                        
                        for task in combined_result.get("tasks", []):
                            status_emoji = "✅" if task["status"] == "completed" else "⏳"
                            task_line = f"{status_emoji} {task['title']}"
                            
                            if task.get("due"):
                                formatted_date = format_task_date(task["due"])
                                task_line += f"\n   📅 Due: {formatted_date}"
                            
                            if task.get("notes"):
                                task_line += f"\n   📝 Notes: {task['notes']}"
                            
                            yield f"{task_line}\n\n"
                        
                        # Add separator if we also have events
                        if has_events:
                            yield "---\n\n"
                    
                    # Display events if we have any
                    if has_events:
                        yield "📅 **EVENTS/MEETINGS**\n\n"
                        
                        for event in combined_result.get("events", []):
                            event_title = event.get("title", "Untitled Event")
                            event_line = f"🗓️ {event_title}"
                            
                            # Format start and end times
                            start_time = event.get("start", "")
                            end_time = event.get("end", "")
                            
                            if start_time:
                                formatted_start = format_task_date(start_time)
                                if event.get("is_all_day"):
                                    event_line += f"\n   ⏰ When: {formatted_start} (All day)"
                                else:
                                    formatted_end = format_task_date(end_time)
                                    event_line += f"\n   ⏰ When: {formatted_start} to {formatted_end}"
                            
                            if event.get("location"):
                                event_line += f"\n   📍 Location: {event['location']}"
                            
                            if event.get("meet_link"):
                                event_line += f"\n   🔗 Meet: {event['meet_link']}"
                            
                            if event.get("link"):
                                event_line += f"\n   🌐 Calendar: {event['link']}"
                            
                            yield f"{event_line}\n\n"
                else:
                    # Handle errors
                    if combined_result.get("tasks_error") and combined_result.get("events_error"):
                        yield f"❌ Failed to retrieve schedule: Tasks error - {combined_result.get('tasks_error')}, Events error - {combined_result.get('events_error')}"
                    elif combined_result.get("tasks_error"):
                        yield f"❌ Failed to retrieve tasks: {combined_result.get('tasks_error')}\n\n"
                        if combined_result.get("events"):
                            # Still show events if we got them
                            yield "📅 **EVENTS**\n\n"
                            # (Event display code would go here - same as above)
                    elif combined_result.get("events_error"):
                        yield f"❌ Failed to retrieve events: {combined_result.get('events_error')}\n\n"
                        if combined_result.get("tasks"):
                            # Still show tasks if we got them
                            yield "🗒️ **TASKS**\n\n"
                            # (Task display code would go here - same as above)
                    else:
                        yield "❌ Failed to retrieve your schedule: Unknown error"
                return
            
            # Handle regular task requests
            elif is_get_tasks or is_time_specific_tasks:
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
            
            # Handle event requests
            elif is_get_events or is_time_specific_events:
                logger.info("Processing event retrieval request")
                events_result = await self._get_events(content)
                
                if events_result.get("success"):
                    yield "📅 Here are your events:\n\n"
                    
                    if not events_result.get("events"):
                        yield "No events found for your request. You can create new events by saying something like 'schedule a meeting for tomorrow at 2pm'.\n\n"
                        return
                    
                    for event in events_result.get("events", []):
                        event_title = event.get("title", "Untitled Event")
                        event_line = f"🗓️ {event_title}"
                        
                        # Format start and end times
                        start_time = event.get("start", "")
                        end_time = event.get("end", "")
                        
                        if start_time:
                            formatted_start = format_task_date(start_time)
                            if event.get("is_all_day"):
                                event_line += f"\n   ⏰ When: {formatted_start} (All day)"
                            else:
                                formatted_end = format_task_date(end_time)
                                event_line += f"\n   ⏰ When: {formatted_start} to {formatted_end}"
                        
                        if event.get("location"):
                            event_line += f"\n   📍 Location: {event['location']}"
                        
                        if event.get("description"):
                            event_line += f"\n   📝 Description: {event['description']}"
                        
                        if event.get("meet_link"):
                            event_line += f"\n   🔗 Meet: {event['meet_link']}"
                        
                        if event.get("link"):
                            event_line += f"\n   🌐 Calendar: {event['link']}"
                        
                        # Show attendees if present
                        if event.get("attendees"):
                            attendees_count = len(event["attendees"])
                            if attendees_count > 0:
                                event_line += f"\n   👥 Attendees: {attendees_count} people"
                        
                        yield f"{event_line}\n\n"
                else:
                    yield f"❌ Failed to retrieve events: {events_result.get('error', 'Unknown error')}"
                return

            # Check for event keywords (only for creation, retrieval is handled above)
            event_keywords = ["meeting", "schedule meeting", "create meeting", "set meeting", 
                            "event", "create event", "set event", "calendar event",
                            "appointment", "schedule", "interview", "call", "conference",
                            "webinar", "session", "catch up", "sync", "discussion"]
            is_event = any(keyword in content_lower for keyword in event_keywords) and not (is_get_events or is_time_specific_events)
            
            # Check for task/reminder keywords first (after checking for retrievals)
            task_keywords = ["reminder", "remind me", "create task", "set task", "set reminder", 
                           "create reminder", "todo", "task"]
            is_task = any(keyword in content_lower for keyword in task_keywords) and not (is_get_tasks or is_time_specific_tasks)
            
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
            
            # Handle event/meeting request - after checking for retrievals
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
