from typing import Optional, ClassVar
import logging
import requests
import json
from langchain.tools import BaseTool
from datetime import datetime

logger = logging.getLogger(__name__)

class GetTasksTool(BaseTool):
    """Tool for getting tasks from Google Tasks"""
    name: ClassVar[str] = "get_tasks"
    description: ClassVar[str] = """Gets tasks from Google Tasks.
    Input should be a JSON string with optional fields:
    - today_only: If true, only returns today's tasks (default: false)
    
    Example: {"today_only": true}
    """
    
    def __init__(self, access_token: str):
        super().__init__()
        self.access_token = access_token
        self.api_url = "https://tasks.googleapis.com/tasks/v1"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    
    def _run(self, query: str) -> str:
        """Execute the task retrieval"""
        try:
            # Parse input
            params = json.loads(query) if query else {}
            today_only = params.get("today_only", False)
            
            # Get task lists
            lists_response = requests.get(f"{self.api_url}/users/@me/lists", headers=self.headers)
            lists_response.raise_for_status()
            task_lists = lists_response.json().get("items", [])
            
            if not task_lists:
                return json.dumps({
                    "success": True,
                    "message": "No task lists found",
                    "tasks": []
                })
            
            # Get tasks from first list
            task_list_id = task_lists[0]["id"]
            tasks_response = requests.get(
                f"{self.api_url}/lists/{task_list_id}/tasks",
                headers=self.headers
            )
            tasks_response.raise_for_status()
            tasks = tasks_response.json().get("items", [])
            
            # Filter for today's tasks if requested
            if today_only:
                today = datetime.now().strftime("%Y-%m-%d")
                tasks = [
                    task for task in tasks 
                    if task.get("due") and task.get("due").startswith(today)
                ]
            
            # Format tasks for display
            formatted_tasks = []
            for task in tasks:
                formatted_task = {
                    "title": task.get("title"),
                    "status": task.get("status", "needsAction"),
                    "due": task.get("due"),
                    "notes": task.get("notes")
                }
                formatted_tasks.append(formatted_task)
            
            return json.dumps({
                "success": True,
                "task_list": task_lists[0]["title"],
                "tasks": formatted_tasks
            })
            
        except Exception as e:
            logger.error(f"Error getting tasks: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    async def _arun(self, query: str) -> str:
        """Execute the task retrieval asynchronously"""
        return self._run(query) 