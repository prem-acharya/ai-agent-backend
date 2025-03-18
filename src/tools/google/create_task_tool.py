from typing import Optional, ClassVar
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
    
    Example: {"title": "Buy groceries", "due": "tomorrow", "notes": "Get milk and eggs"}
    """
    
    def __init__(self, access_token: str):
        super().__init__()
        self.access_token = access_token
        self.api_url = "https://tasks.googleapis.com/tasks/v1"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    
    def _format_due_date(self, due: str) -> str:
        """Format the due date to Google Tasks API format"""
        if due.lower() == "today":
            return datetime.now().strftime("%Y-%m-%dT00:00:00.000Z")
        elif due.lower() == "tomorrow":
            tomorrow = datetime.now() + timedelta(days=1)
            return tomorrow.strftime("%Y-%m-%dT00:00:00.000Z")
        elif len(due) == 10:  # YYYY-MM-DD format
            return f"{due}T00:00:00.000Z"
        return due
    
    def _run(self, query: str) -> str:
        """Execute the task creation"""
        try:
            # Parse input
            task_data = json.loads(query)
            title = task_data.get("title")
            if not title:
                return "Error: Task title is required"
            
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
            
            # Prepare task data
            task_body = {"title": title}
            if "due" in task_data:
                task_body["due"] = self._format_due_date(task_data["due"])
            if "notes" in task_data:
                task_body["notes"] = task_data["notes"]
            
            # Create task
            create_response = requests.post(
                f"{self.api_url}/lists/{task_list_id}/tasks",
                headers=self.headers,
                json=task_body
            )
            create_response.raise_for_status()
            
            return json.dumps({
                "success": True,
                "message": f"Task '{title}' created successfully",
                "task": create_response.json()
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