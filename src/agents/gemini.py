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
                if task_data:
                    for tool in self.tools:
                        if tool.name == "create_task":
                            result_json = await tool._arun(json.dumps(task_data))
                            try:
                                result_data = json.loads(result_json)
                                if result_data.get("success"):
                                    notes = task_data.get('notes', 'N/A').replace("\n", "\n\n")
                                    task_details = f"Title: {task_data.get('title', 'N/A')}\n\nDue: {task_data.get('due', 'N/A')}\n\nNotes: {notes}"
                                    yield f"Task/Reminder created successfully!\n\n\nTask Details:\n\n\n{task_details}"
                                else:
                                    yield f"❌ Failed to create task: {result_data.get('error')}"
                            except Exception as e:
                                yield f"❌ Error processing task creation: {str(e)}"
                return
            
            # Check for event/meeting keywords
            event_keywords = ["meeting", "schedule meeting", "create meeting", "set meeting", 
                            "event", "create event", "set event", "calendar event"]
            is_event = any(keyword in content_lower for keyword in event_keywords)
            
            # Handle event/meeting request
            if is_event:
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

    async def _prepare_task_data(self, content: str) -> dict:
        """Prepare task data from user input using AI analysis"""
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
        
        # Clean up title by removing time information
        time_patterns = [
            r'\s*at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)',  # at 2 PM
            r'\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)',       # 2 PM
            r'\s*\d{2}:\d{2}'                           # 14:00
        ]
        for pattern in time_patterns:
            task_title = re.sub(pattern, '', task_title, flags=re.IGNORECASE)
        
        # Use Gemini to analyze the task and generate a detailed description
        analysis_prompt = f"""You are a task analysis AI. Analyze this task request and provide a detailed task description with relevant emojis.
User Request: {content}

IMPORTANT: Respond ONLY with a valid JSON object in the following format:
{{
    "title": "Clear, action-oriented title with relevant emoji (DO NOT include due dates like 'tomorrow' or 'today' in title)",
    "description": "Detailed description with relevant emojis explaining what needs to be done",
    "due": "today",
    "notes": "2 to 3 concise bullet points with relevant emojis (make sure to not include more than 3 points)"
}}

Consider these aspects when generating the response:
1. Title: Make it clear and actionable with a relevant emoji. NEVER include time words like 'today', 'tomorrow', 'next week' in the title.
2. Description: Break down the task steps with appropriate emojis
3. Due Date: Use "today" as default unless a specific date is mentioned in the user request
4. Notes: actionable bullet points with relevant emojis (make sure to not include more than 3 points)

Example response:
{{
    "title": "📋 Comprehensive Document Review",
    "description": "🔍 Review all documents thoroughly\n📝 Check for completeness and accuracy\n✍️ Note required changes\n✅ Create summary",
    "due": "today",
    "notes": "⏰ Estimated: 2-3 hours\n📌 Priority: High\n💡 Start with executive summary"
}}

Now analyze this task and provide your response:"""

        try:
            # Get Gemini's analysis using await
            response = await self.llm.agenerate([[HumanMessage(content=analysis_prompt)]])
            response_text = response.generations[0][0].text.strip()
            
            # Try to extract JSON from the response
            try:
                # First try direct JSON parsing
                analysis = json.loads(response_text)
            except json.JSONDecodeError:
                # If that fails, try to find JSON in the text
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        raise ValueError("Could not parse JSON from response")
                else:
                    raise ValueError("No JSON found in response")
            
            # Validate required fields
            required_fields = ["title", "description", "due", "notes"]
            for field in required_fields:
                if field not in analysis:
                    raise ValueError(f"Missing required field: {field}")
            
            # Use the AI-generated content but clean up the title
            title = analysis["title"]
            # Remove any potential due date references from title
            date_terms = ["today", "tomorrow", "next week", "next month", "next day"]
            for term in date_terms:
                title = re.sub(r'\b' + term + r'\b', '', title, flags=re.IGNORECASE)
            # Cleanup any double spaces created by removals
            title = re.sub(r'\s+', ' ', title).strip()
            task_data["title"] = title
            
            # Set due date to today by default unless explicitly specified
            if "tomorrow" in content_lower:
                task_data["due"] = "tomorrow"
            elif "next week" in content_lower:
                task_data["due"] = "next week"
            elif "next month" in content_lower:
                task_data["due"] = "next month"
            else:
                task_data["due"] = "today"
            
            # Combine notes with AI-generated content
            notes = []
            
            # Add AI-generated description as primary note
            if analysis.get("description"):
                notes.append(analysis["description"])
            
            # Add AI-generated additional notes (limit to 2-5 points)
            if analysis.get("notes"):
                if isinstance(analysis["notes"], str):
                    note_points = analysis["notes"].split("\n")
                elif isinstance(analysis["notes"], list):
                    note_points = analysis["notes"]
                else:
                    note_points = []
                # Take only first 4 points if more exist
                notes.extend(note_points[:4])
            
            # Add time information if present in user input
            time_match = re.search(r'(?:at\s+)?(\d{1,2})(?::\d{2})?\s*(?:am|pm)', content_lower)
            if time_match:
                hour = time_match.group(1)
                if "pm" in time_match.group(0).lower():
                    hour = str(int(hour) + 12) if int(hour) < 12 else hour
                notes.append(f"⏰ Scheduled for {hour}:00")
            
            # Add user's specific notes if provided
            if "notes:" in content_lower:
                user_notes = content.split("notes:", 1)[1].strip()
                notes.append(f"👤 User Notes: {user_notes}")
            
            if notes:
                # Ensure we have at least 2 points and at most 5 points
                if len(notes) < 2:
                    # Add default points if we have less than 2
                    notes.extend([
                        "📌 Priority: Medium",
                    ])
                elif len(notes) > 5:
                    # Take only first 5 points if we have more
                    notes = notes[:5]
                task_data["notes"] = "\n".join(notes)
            
        except Exception as e:
            logger.error(f"Error getting task analysis from Gemini: {str(e)}")
            # Fallback to basic task data if analysis fails
            task_title = task_title.strip().title()
            # Remove any due date terms from title
            date_terms = ["today", "tomorrow", "next week", "next month", "next day"]
            for term in date_terms:
                task_title = re.sub(r'\b' + term + r'\b', '', task_title, flags=re.IGNORECASE)
            # Cleanup any double spaces created by removals
            task_title = re.sub(r'\s+', ' ', task_title).strip()
            task_data["title"] = task_title
            
            # Set proper due date based on content
            if "tomorrow" in content_lower:
                task_data["due"] = "tomorrow"
            elif "next week" in content_lower:
                task_data["due"] = "next week"
            elif "next month" in content_lower:
                task_data["due"] = "next month"
            else:
                task_data["due"] = "today"  # Default to today
            
            # Add basic notes
            notes = []
            time_match = re.search(r'(?:at\s+)?(\d{1,2})(?::\d{2})?\s*(?:am|pm)', content_lower)
            if time_match:
                hour = time_match.group(1)
                if "pm" in time_match.group(0).lower():
                    hour = str(int(hour) + 12) if int(hour) < 12 else hour
                notes.append(f"⏰ Scheduled for {hour}:00")
            
            if "notes:" in content_lower:
                notes.append(f"👤 User Notes: {content.split('notes:', 1)[1].strip()}")
            
            # Ensure we have at least 2 points
            if len(notes) < 2:
                notes.extend([
                    "📌 Priority: Medium",
                    "💡 Take breaks between reviews"
                ])
            
            if notes:
                task_data["notes"] = "\n".join(notes[:5])  # Limit to 5 points
        
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
        time_patterns = [
            r'(\d{1,2})(?::\d{2})?\s*(?:am|pm)\s*to\s*(\d{1,2})(?::\d{2})?\s*(?:am|pm)',  # 5pm to 6pm
            r'(\d{1,2}):(\d{2})\s*to\s*(\d{1,2}):(\d{2})',  # 17:00 to 18:00
            r'from\s*(\d{1,2})(?::\d{2})?\s*(?:am|pm)\s*to\s*(\d{1,2})(?::\d{2})?\s*(?:am|pm)'  # from 5pm to 6pm
        ]
        
        for pattern in time_patterns:
            if time_match := re.search(pattern, content_lower):
                if len(time_match.groups()) == 2:  # AM/PM format
                    start_hour, end_hour = time_match.groups()
                    
                    # Convert to 24-hour format
                    if "pm" in content_lower.split("to")[0].lower():
                        start_hour = str(int(start_hour) + 12) if int(start_hour) < 12 else start_hour
                    if "pm" in content_lower.split("to")[1].lower():
                        end_hour = str(int(end_hour) + 12) if int(end_hour) < 12 else end_hour
                    
                    event_data["start_time"] = f"{int(start_hour):02d}:00"
                    event_data["end_time"] = f"{int(end_hour):02d}:00"
                elif len(time_match.groups()) == 4:  # 24-hour format
                    start_hour, start_min, end_hour, end_min = time_match.groups()
                    event_data["start_time"] = f"{int(start_hour):02d}:{start_min}"
                    event_data["end_time"] = f"{int(end_hour):02d}:{end_min}"
                break
        
        # If no time range found, try to find single time and set duration to 1 hour
        if "start_time" not in event_data:
            single_time_pattern = r'(?:at\s+)?(\d{1,2})(?::\d{2})?\s*(?:am|pm)'
            if time_match := re.search(single_time_pattern, content_lower):
                hour = time_match.group(1)
                if "pm" in time_match.group(0).lower():
                    hour = str(int(hour) + 12) if int(hour) < 12 else hour
                event_data["start_time"] = f"{int(hour):02d}:00"
                event_data["end_time"] = f"{(int(hour) + 1):02d}:00"
        
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
