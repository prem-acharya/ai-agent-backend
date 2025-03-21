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
    
    Example: {
        "title": "Daily Hydration Reminder",
        "notes": "Drink water",
        "due": "2025-03-21"
    }
    
    This would create a task with due date "2025-03-21T10:00:00.000Z" (default time is 10:00 AM).
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
    
    def _format_due_datetime(self, due: str) -> str:
        """Format the due date to Google Tasks API format (RFC 3339 timestamp) with fixed 10:00 AM time"""
        if due.lower() == "today":
            date = datetime.now()
        elif due.lower() == "tomorrow":
            date = datetime.now() + timedelta(days=1)
        else:
            try:
                # Try to parse the date in various formats
                if "T" in due and due.endswith("Z"):
                    # Already in RFC 3339 format
                    return due
                elif "T" in due:
                    # Has time but no Z
                    return f"{due}Z"
                else:
                    # Just a date string
                    date = datetime.strptime(due, "%Y-%m-%d")
            except ValueError:
                # If can't parse, use default format with current date
                logger.warning(f"Could not parse date: {due}, using today with default time")
                date = datetime.now()

        # Always use 10:00 AM time
        date = date.replace(hour=10, minute=0, second=0, microsecond=0)

        # Format according to RFC 3339 timestamp format required by Google Tasks
        # Format: YYYY-MM-DDTHH:MM:SS.000Z
        rfc3339_format = f"{date.strftime('%Y-%m-%d')}T10:00:00.000Z"
        logger.info(f"Formatted date to RFC 3339: {rfc3339_format}")
        return rfc3339_format

    def _run(self, query: str) -> str:
        """Execute the task creation"""
        try:
            # Parse input
            task_data = json.loads(query)
            logger.info(f"Received task data: {json.dumps(task_data, indent=2)}")
            
            # Validate and set defaults
            if not task_data.get("title"):
                logger.warning("Task title is missing")
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
            
            # Build task body
            task_body = {
                "title": task_data["title"],
                "status": "needsAction"
            }
            
            # Handle due date (always with 10:00 AM time)
            if "due" in task_data:
                due_date = self._format_due_datetime(task_data["due"])
                task_body["due"] = due_date
                logger.info(f"Formatted due date: {due_date}")
            
            # Handle notes
            if "notes" in task_data:
                task_body["notes"] = task_data["notes"]
            
            # Log the final task body for debugging
            logger.info(f"Final task_body: {json.dumps(task_body, indent=2)}")
            
            # Verify essential fields
            if "due" in task_body:
                if not (task_body["due"].endswith("Z") and "T" in task_body["due"]):
                    logger.warning(f"Due date may not be properly formatted: {task_body['due']}")
                    # Fix the format if needed
                    if not "T" in task_body["due"]:
                        task_body["due"] = f"{task_body['due']}T10:00:00.000Z"
            
            # Create task
            create_response = requests.post(
                f"{self.api_url}/lists/{task_list_id}/tasks",
                headers=self.headers,
                json=task_body
            )
            create_response.raise_for_status()
            
            created_task = create_response.json()
            logger.info(f"Task created successfully: {json.dumps(created_task, indent=2)}")
            
            return json.dumps({
                "success": True,
                "message": f"Task '{original_title}' created successfully",
                "task": {
                    "title": created_task.get("title"),
                    "due": created_task.get("due"),
                    "notes": created_task.get("notes"),
                    "status": created_task.get("status")
                },
                "request_details": {
                    "due": task_body.get("due")
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