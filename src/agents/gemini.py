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

    def _get_topic_emoji(self, topic: str) -> str:
        """Get relevant emoji for the topic."""
        topic_lower = topic.lower()
        if any(word in topic_lower for word in ["ai", "machine learning", "ml", "artificial intelligence"]):
            return "🤖"
        elif any(word in topic_lower for word in ["program", "code", "develop"]):
            return "💻"
        elif any(word in topic_lower for word in ["learn", "study", "practice"]):
            return "📚"
        elif any(word in topic_lower for word in ["research", "analyze", "investigate"]):
            return "🔍"
        elif any(word in topic_lower for word in ["design", "ui", "ux", "interface"]):
            return "🎨"
        elif any(word in topic_lower for word in ["plan", "schedule", "organize"]):
            return "📅"
        elif any(word in topic_lower for word in ["test", "debug", "fix"]):
            return "🧪"
        elif any(word in topic_lower for word in ["document", "write", "note"]):
            return "📝"
        elif any(word in topic_lower for word in ["meet", "discuss", "collaborate"]):
            return "👥"
        elif any(word in topic_lower for word in ["automate", "automation"]):
            return "⚙️"
        return "✅"  # Default emoji

    async def generate_response(self, content: str) -> AsyncIterable[str]:
        """Generate a response using the agent."""
        try:
            await self.reset_callback()
            
            # Handle websearch if enabled
            if self.websearch:
                # First get current time
                try:
                    from src.tools.datetime.time_tool import CurrentTimeTool
                    from src.tools.websearch.websearch_tool import WebSearchTool
                    
                    time_tool = CurrentTimeTool()
                    current_time = await time_tool._arun()
                    
                    # Then perform web search
                    yield "Searching the web\n\n"
                    web_tool = WebSearchTool()
                    web_results = await web_tool._arun(content)
                    
                    # Combine context
                    content = f"""
Current Time: {current_time}
User Question: {content}
Web Search Results: {web_results}
"""
                except Exception as e:
                    logger.error(f"Error during websearch: {str(e)}")
                    # Continue with original content if web search fails
            
            # Check if this is a complex query (info + task)
            is_complex_query = any(word in content.lower() for word in ["remind", "reminder", "task"]) and "what" in content.lower()
            
            if is_complex_query:
                # First, get information about the topic
                info_query = content.lower().split("remind")[0].strip()
                if info_query.endswith("and"):
                    info_query = info_query[:-3].strip()
                
                # Get information response
                response = await self.llm.agenerate([[HumanMessage(content=info_query)]])
                info_response = response.generations[0][0].text
                yield f"📚 Information:\n\n{info_response}\n\n"
                
                # Now create the task
                task_data = {}
                
                # Extract topic for task title
                topic = info_query.replace("what is", "").replace("what are", "").strip()
                topic_emoji = self._get_topic_emoji(topic)
                task_data["title"] = f"{topic_emoji} Learn about {topic.title()}"  # Capitalize the topic
                
                # Extract due date dynamically from user input
                content_lower = content.lower()
                due_date = "today"  # Default value
                
                # Look for time/date patterns in the content
                if any(day in content_lower for day in ["tomorrow", "tmr"]):
                    due_date = "tomorrow"
                elif "next week" in content_lower:
                    from datetime import datetime, timedelta
                    due_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
                elif "today" in content_lower:
                    due_date = "today"
                elif any(pattern in content_lower for pattern in ["next month", "in a month"]):
                    from datetime import datetime, timedelta
                    due_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
                else:
                    # Try to find a specific date in the format "YYYY-MM-DD" or similar patterns
                    import re
                    date_patterns = [
                        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                        r'\d{2}-\d{2}-\d{4}'   # DD-MM-YYYY
                    ]
                    for pattern in date_patterns:
                        if match := re.search(pattern, content):
                            due_date = match.group(0)
                            break
                
                task_data["due"] = due_date
                
                # Format notes with sorted sections
                key_points = []
                action_items = []
                
                # Extract key points from info_response
                for line in info_response.split('\n'):
                    if line.strip() and not line.startswith('#') and not line.startswith('*'):
                        key_points.append(line.strip())
                
                # Standard action items for learning tasks
                action_items = [
                    "Study and understand core concepts",
                    "Research practical applications",
                    "Practice with hands-on examples",
                    "Review and validate understanding",
                    "Document key learnings and insights"
                ]
                
                # Format notes with emojis and sections
                task_data["notes"] = (
                    "🎯 Key Learning Points:\n" + 
                    "\n".join(f"• {point}" for point in random.sample(key_points, min(len(key_points), 6))) +
                    "\n\n📝 Action Items:\n" + 
                    "\n".join(f"• {item}" for item in action_items)
                )
                
                # Create task using the tool
                if self.google_access_token:
                    for tool in self.tools:
                        if tool.name == "create_task":
                            result_json = await tool._arun(json.dumps(task_data))
                            try:
                                result_data = json.loads(result_json)
                                if result_data.get("success"):
                                    yield f"\n\n✨ Task created successfully!\n\nDetails:\n```json\n{json.dumps(task_data, indent=2)}\n```"
                                else:
                                    yield f"\n\n❌ Failed to create task: {result_data.get('error')}"
                            except Exception as e:
                                yield f"\n\n❌ Error processing task creation: {str(e)}"
                            return
            
            # Handle regular task creation
            elif self.google_access_token and any(word in content.lower() for word in ["task", "todo", "reminder", "set"]):
                logger.info(f"Processing task-related request: {content}")
                try:
                    # Parse the task request
                    task_data = {}
                    
                    # Extract task title and get context
                    content_lower = content.lower()
                    if "to " in content_lower:
                        task_title = content_lower.split("to ", 1)[1].strip()
                    else:
                        task_title = content_lower.replace("task", "").replace("set", "").strip()
                    
                    # Get task context from AI
                    context_prompt = f"""Generate a task details in this exact JSON format:
{{
    "title": "A clear, concise title for the task",
    "notes": ["3-4 bullet points as an array about the task"]
}}
For this task: {task_title}"""
                    
                    response = await self.llm.agenerate([[HumanMessage(content=context_prompt)]])
                    try:
                        # Extract JSON from the response text
                        import re
                        json_match = re.search(r'```(?:json)?\s*({[\s\S]+?})\s*```', response.generations[0][0].text)
                        if json_match:
                            ai_response = json.loads(json_match.group(1))
                        else:
                            ai_response = json.loads(response.generations[0][0].text)
                        
                        # Get appropriate emoji based on the AI-generated title
                        topic_emoji = self._get_topic_emoji(ai_response["title"])
                        
                        # Set task data using AI response
                        task_data["title"] = f"{topic_emoji} {ai_response['title']}"
                        
                        # Format notes as bullet points
                        if isinstance(ai_response["notes"], list):
                            task_data["notes"] = "📝 Task Details:\n" + "\n".join(f"• {note}" for note in ai_response["notes"])
                        else:
                            task_data["notes"] = f"📝 Task Details:\n• {ai_response['notes']}"
                            
                    except (json.JSONDecodeError, KeyError) as e:
                        # Fallback if JSON parsing fails
                        logger.error(f"Error parsing AI response: {str(e)}")
                        task_data["title"] = f"✅ {task_title.title()}"
                        task_data["notes"] = f"📝 Task Details:\n• {response.generations[0][0].text}"
                    
                    # Extract due date dynamically from user input
                    content_lower = content.lower()
                    due_date = "today"  # Default value
                    
                    # Look for time/date patterns in the content
                    if any(day in content_lower for day in ["tomorrow", "tmr"]):
                        due_date = "tomorrow"
                    elif "next week" in content_lower:
                        from datetime import datetime, timedelta
                        due_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
                    elif "today" in content_lower:
                        due_date = "today"
                    elif any(pattern in content_lower for pattern in ["next month", "in a month"]):
                        from datetime import datetime, timedelta
                        due_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
                    else:
                        # Try to find a specific date in the format "YYYY-MM-DD" or similar patterns
                        import re
                        date_patterns = [
                            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                            r'\d{2}-\d{2}-\d{4}'   # DD-MM-YYYY
                        ]
                        for pattern in date_patterns:
                            if match := re.search(pattern, content):
                                due_date = match.group(0)
                                break
                    
                    task_data["due"] = due_date
                    
                    # Create task using the tool
                    for tool in self.tools:
                        if tool.name == "create_task":
                            result_json = await tool._arun(json.dumps(task_data))
                            try:
                                result_data = json.loads(result_json)
                                if result_data.get("success"):
                                    yield f"✨ Task created successfully!\n\nDetails:\n```json\n{json.dumps(task_data, indent=2)}\n```"
                                else:
                                    yield f"❌ Failed to create task: {result_data.get('error')}"
                            except Exception as e:
                                yield f"❌ Error processing task creation: {str(e)}"
                            return
                            
                except Exception as e:
                    logger.exception(f"Error handling task request: {str(e)}")
                    yield f"⚠️ I encountered an error while processing your task request: {str(e)}"
            else:
                # For non-task queries, use direct response or reasoning based on the setting
                if self.reasoning:
                    yield "reasoning start\n\n"
                    # Use Gemini model with reasoning
                    response = await self.llm.agenerate([[HumanMessage(content=f"Think step by step to answer this question: {content}")]])
                    reasoning_text = response.generations[0][0].text
                    yield reasoning_text
                    
                    yield "\n\nFinal Answer start\n\n"
                    # Get a final summarized answer
                    summary_prompt = f"Based on the above reasoning, provide a concise final answer to the original question: {content}"
                    response = await self.llm.agenerate([[HumanMessage(content=summary_prompt)]])
                    yield response.generations[0][0].text
                else:
                    # Direct response
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
