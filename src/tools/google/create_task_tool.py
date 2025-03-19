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
    - time: Time in 24-hour format "HH:MM" (optional)
    - repeat: Repeat settings (optional, defaults to none) with fields:
      - frequency: "daily", "weekly", "monthly", or "yearly"
      - count: Number of occurrences (optional)
      - until: End date in "YYYY-MM-DD" format (optional)
    
    Example: {
        "title": "Team Meeting",
        "due": "2024-03-19",
        "time": "14:30",
        "notes": "Discuss project progress"
    }
    """
    access_token: str
    api_url: str = "https://tasks.googleapis.com/tasks/v1"
    headers: dict = None
    calendar_api_url: str = "https://www.googleapis.com/calendar/v3"
    
    def __init__(self, access_token: str):
        super().__init__(access_token=access_token)
        self.api_url = "https://tasks.googleapis.com/tasks/v1"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    
    def _format_due_datetime(self, due: str, time: Optional[str] = None) -> str:
        """Format the due date and time to Google Tasks API format"""
        # Handle date
        if due.lower() == "today":
            date = datetime.now()
        elif due.lower() == "tomorrow":
            date = datetime.now() + timedelta(days=1)
        else:
            try:
                date = datetime.strptime(due, "%Y-%m-%d")
            except ValueError:
                return due

        # Add time if provided
        if time:
            try:
                hours, minutes = map(int, time.split(":"))
                date = date.replace(hour=hours, minute=minutes, second=0, microsecond=0)
            except (ValueError, TypeError):
                logger.warning(f"Invalid time format: {time}, using default time")
                date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            date = date.replace(hour=0, minute=0, second=0, microsecond=0)

        return date.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    def _create_calendar_event(self, task_data: dict) -> dict:
        """Create a Calendar event for tasks with time or recurrence"""
        try:
            # Format event data
            event_body = {
                "summary": task_data["title"],
                "description": task_data.get("notes", "")
            }
            
            # Add attendees if provided
            if "attendees" in task_data and task_data["attendees"]:
                event_body["attendees"] = [
                    {"email": email.strip()} 
                    for email in task_data["attendees"]
                ]
            
            # Add organizer if provided
            if "organizer" in task_data and task_data["organizer"]:
                event_body["organizer"] = {"email": task_data["organizer"]}
            
            # Set start and end times
            date_str = task_data.get("due", "today")
            time_str = task_data.get("time", "00:00")
            
            if date_str.lower() == "today":
                start_date = datetime.now()
            elif date_str.lower() == "tomorrow":
                start_date = datetime.now() + timedelta(days=1)
            else:
                try:
                    start_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    start_date = datetime.now()
            
            # Add time if provided
            try:
                hours, minutes = map(int, time_str.split(":"))
                start_date = start_date.replace(hour=hours, minute=minutes, second=0, microsecond=0)
            except (ValueError, TypeError):
                start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # End time is 1 hour after start time by default
            end_date = start_date + timedelta(hours=1)
            
            # Format start and end times
            event_body["start"] = {
                "dateTime": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "timeZone": "Asia/Kolkata"  # Using IST timezone
            }
            event_body["end"] = {
                "dateTime": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "timeZone": "Asia/Kolkata"  # Using IST timezone
            }
            
            # Add recurrence only if explicitly provided and valid
            if "repeat" in task_data and task_data["repeat"] and isinstance(task_data["repeat"], dict):
                repeat_data = task_data["repeat"]
                frequency = repeat_data.get("frequency", "").upper()
                
                if frequency in ["DAILY", "WEEKLY", "MONTHLY", "YEARLY"]:
                    recurrence = [f"RRULE:FREQ={frequency}"]
                    
                    # Add count if available
                    if "count" in repeat_data and repeat_data["count"]:
                        try:
                            count = int(repeat_data["count"])
                            if count > 0:
                                recurrence[0] += f";COUNT={count}"
                        except (ValueError, TypeError):
                            pass
                    
                    # Add until if available
                    if "until" in repeat_data and repeat_data["until"]:
                        try:
                            until_date = datetime.strptime(repeat_data["until"], "%Y-%m-%d")
                            recurrence[0] += f";UNTIL={until_date.strftime('%Y%m%dT235959Z')}"
                        except ValueError:
                            pass
                    
                    event_body["recurrence"] = recurrence
            
            # Create event
            response = requests.post(
                f"{self.calendar_api_url}/calendars/primary/events",
                headers=self.headers,
                json=event_body
            )
            response.raise_for_status()
            event_data = response.json()
            
            return {
                "id": event_data.get("id", ""),
                "htmlLink": event_data.get("htmlLink", "")
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Calendar API request error: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Error creating calendar event: {str(e)}")
            return {}

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
            
            # Check if we need to use calendar (for time or recurrence)
            calendar_event = None
            if ("time" in task_data and task_data["time"]) or ("repeat" in task_data and task_data["repeat"]):
                # Create calendar event
                calendar_event = self._create_calendar_event(task_data)
                
                # Add calendar link to notes
                if calendar_event and calendar_event.get("htmlLink"):
                    calendar_note = f"\n\n📅 View in Calendar: {calendar_event['htmlLink']}"
                    if "notes" in task_data:
                        task_data["notes"] += calendar_note
                    else:
                        task_data["notes"] = calendar_note
            
            # Prepare task data for Google Tasks
            task_body = {
                "title": task_data["title"],
                "status": "needsAction"
            }
            
            # Handle due date (without time for Google Tasks)
            if "due" in task_data:
                task_body["due"] = self._format_due_datetime(task_data["due"])
            
            # Handle notes
            if "notes" in task_data:
                task_body["notes"] = task_data["notes"]
            
            # Create task
            create_response = requests.post(
                f"{self.api_url}/lists/{task_list_id}/tasks",
                headers=self.headers,
                json=task_body
            )
            create_response.raise_for_status()
            
            created_task = create_response.json()
            
            # Prepare response
            response_data = {
                "success": True,
                "message": f"Task '{task_data['title']}' created successfully",
                "task": {
                    "title": created_task.get("title"),
                    "due": created_task.get("due"),
                    "notes": created_task.get("notes"),
                    "status": created_task.get("status")
                }
            }
            
            # Add calendar info if used
            if calendar_event:
                response_data["calendar_event_created"] = True
                response_data["message"] += " with calendar event for time/recurrence"
                response_data["calendar_link"] = calendar_event.get("htmlLink", "")
            
            return json.dumps(response_data)
            
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