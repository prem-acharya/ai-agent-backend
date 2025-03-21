from typing import Optional, ClassVar, Dict, Any
import logging
import requests
import json
from langchain.tools import BaseTool
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CreateTaskTool(BaseTool):
    """Tool for creating tasks in Google Tasks"""
    name: ClassVar[str] = "create_task"
    description: ClassVar[str] = """Creates a new task in Google Tasks.
    Input should be a JSON string with required fields:
    - title: The task title (required)
    - due: Due date (optional, can be "today", "tomorrow", or "YYYY-MM-DD")
    - notes: Additional notes (optional)
    - repeat: Repeat settings (optional, format: {"count": number})
    - time: Time for the task (optional, format: "HH:MM", default: "10:00")
    
    Example: {
        "title": "Complete Project Report",
        "due": "2025-03-21T10:00:00.000Z",
        "notes": "Include all project milestones",
        "recurrence": [
            "RRULE:FREQ=DAILY;COUNT=1"
        ],
    }
    """
    access_token: str
    api_url: str = "https://tasks.googleapis.com/tasks/v1"
    headers: dict = None
    
    def __init__(self, access_token: str):
        super().__init__(access_token=access_token)
        self.api_url = "https://tasks.googleapis.com/tasks/v1"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    
    def _format_due_datetime(self, due: str, time: Optional[str] = None) -> str:
        """Format the due date to Google Tasks API format"""
        if due.lower() == "today":
            date = datetime.now()
        elif due.lower() == "tomorrow":
            date = datetime.now() + timedelta(days=1)
        else:
            try:
                date = datetime.strptime(due, "%Y-%m-%d")
            except ValueError:
                # Try to return as is if it's already formatted
                if "T" in due and "Z" in due:
                    return due
                return f"{due}T10:00:00.000Z"  # Default time

        # Add time if provided, otherwise use default 10:00 AM
        time = time or "10:00"
        try:
            time_obj = datetime.strptime(time, "%H:%M")
            date = date.replace(hour=time_obj.hour, minute=time_obj.minute, second=0, microsecond=0)
        except ValueError:
            date = date.replace(hour=10, minute=0, second=0, microsecond=0)

        # Format according to RFC 3339 timestamp format required by Google Tasks
        return f"{date.strftime('%Y-%m-%d')}T{date.strftime('%H:%M')}:00.000Z"
    
    def _get_recurrence_string(self, repeat_data: Dict[str, Any]) -> str:
        """Convert repeat data to Google Tasks recurrence format (RRULE)"""
        # Always use DAILY frequency with a COUNT
        count = repeat_data.get("count", 10)  # Default to 10 occurrences if not specified
        
        # Create simple RRULE string with COUNT
        return f"RRULE:FREQ=DAILY;COUNT={count}"

    def _run(self, query: str) -> str:
        """Execute the task creation"""
        try:
            # Parse input
            task_data = json.loads(query)
            
            # Validate and set defaults
            if not task_data.get("title"):
                return json.dumps({
                    "success": False,
                    "error": "Task title is required"
                })
            
            # Get or create default task list
            lists_response = requests.get(f"{self.api_url}/users/@me/lists", headers=self.headers)
            lists_response.raise_for_status()
            task_lists = lists_response.json().get("items", [])
            
            if not task_lists:
                # Create default task list
                create_list_response = requests.post(
                    f"{self.api_url}/users/@me/lists",
                    headers=self.headers,
                    json={"title": "AI Assistant Tasks"}
                )
                create_list_response.raise_for_status()
                task_list_id = create_list_response.json()["id"]
            else:
                task_list_id = task_lists[0]["id"]
            
            # Prepare task data for Google Tasks
            original_title = task_data["title"]
            
            # If repeat settings exist, include them in the title
            if task_data.get("repeat"):
                repeat = task_data["repeat"]
                count = repeat.get("count", 1)
                task_data["title"] = f"{original_title} 🔄 Repeats {count} times"
            
            # Build task body
            task_body = {
                "title": task_data["title"],
                "status": "needsAction"
            }
            
            # Handle due date and time
            if "due" in task_data:
                task_body["due"] = self._format_due_datetime(task_data["due"], task_data.get("time"))
            
            # Handle notes
            notes = []
            if "notes" in task_data:
                notes.append(task_data["notes"])
            
            # Handle repeat settings
            if "repeat" in task_data:
                # Add recurrence to task
                repeat_data = task_data["repeat"]
                recurrence_rule = self._get_recurrence_string(repeat_data)
                
                # Google Tasks API expects an array of strings for recurrence
                task_body["recurrence"] = [recurrence_rule]
                
                # Also add to notes for visibility
                count = repeat_data.get("count", 1)
                repeat_note = f"🔄 Repeats {count} times"
                notes.append(repeat_note)
            
            # Handle time settings
            if "time" in task_data:
                time_note = f"⏰ Set for {task_data['time']}"
                notes.append(time_note)
            
            # Combine all notes
            if notes:
                task_body["notes"] = "\n\n".join(notes)
            
            # Create task
            create_response = requests.post(
                f"{self.api_url}/lists/{task_list_id}/tasks",
                headers=self.headers,
                json=task_body
            )
            create_response.raise_for_status()
            
            created_task = create_response.json()
            
            return json.dumps({
                "success": True,
                "message": f"Task '{original_title}' created successfully",
                "task": {
                    "title": created_task.get("title"),
                    "due": created_task.get("due"),
                    "notes": created_task.get("notes"),
                    "status": created_task.get("status"),
                    "recurrence": created_task.get("recurrence")
                }
            })
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON input")
            return json.dumps({
                "success": False,
                "error": "Invalid task data format"
            })
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            return json.dumps({
                "success": False,
                "error": f"Failed to communicate with API: {str(e)}"
            })
        except Exception as e:
            logger.error(f"Error creating task: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    async def _arun(self, query: str) -> str:
        """Execute the task creation asynchronously"""
        return self._run(query) 