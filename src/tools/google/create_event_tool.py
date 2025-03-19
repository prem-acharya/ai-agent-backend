from typing import Optional, ClassVar
import logging
import requests
import json
from langchain.tools import BaseTool
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CreateEventTool(BaseTool):
    """Tool for creating events/meetings in Google Calendar"""
    name: ClassVar[str] = "create_event"
    description: ClassVar[str] = """Creates a new event/meeting in Google Calendar.
    Input should be a JSON string with required fields:
    - title: The event title (required)
    - start_time: Start time in 24-hour format "HH:MM" (required)
    - end_time: End time in 24-hour format "HH:MM" (required)
    - due: Event date (required, can be "today", "tomorrow", or "YYYY-MM-DD")
    - notes: Additional notes/description (optional)
    - attendees: List of attendee email addresses (optional)
    - repeat: Repeat settings (optional) with fields:
      - frequency: "daily", "weekly", "monthly", or "yearly"
      - count: Number of occurrences (optional)
      - until: End date in "YYYY-MM-DD" format (optional)
    
    Example: {
        "title": "Team Meeting",
        "start_time": "14:30",
        "end_time": "15:30",
        "due": "2024-03-19",
        "notes": "Discuss project progress",
        "attendees": ["user@example.com"]
    }
    """
    access_token: str
    calendar_api_url: str = "https://www.googleapis.com/calendar/v3"
    
    def __init__(self, access_token: str):
        super().__init__(access_token=access_token)
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

    def _create_calendar_event(self, event_data: dict) -> dict:
        """Create a Calendar event"""
        try:
            # Format event data
            event_body = {
                "summary": event_data["title"],
                "description": event_data.get("notes", "")
            }
            
            # Add attendees if provided
            if "attendees" in event_data and event_data["attendees"]:
                event_body["attendees"] = [
                    {"email": email.strip()} 
                    for email in event_data["attendees"]
                ]
            
            # Set start and end times
            date_str = event_data.get("due", "today")
            start_time = event_data.get("start_time", "00:00")
            end_time = event_data.get("end_time")
            
            if date_str.lower() == "today":
                start_date = datetime.now()
            elif date_str.lower() == "tomorrow":
                start_date = datetime.now() + timedelta(days=1)
            else:
                try:
                    start_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    start_date = datetime.now()
            
            # Add start time
            try:
                hours, minutes = map(int, start_time.split(":"))
                start_date = start_date.replace(hour=hours, minute=minutes, second=0, microsecond=0)
            except (ValueError, TypeError):
                start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Set end time
            if end_time:
                try:
                    end_hours, end_minutes = map(int, end_time.split(":"))
                    end_date = start_date.replace(hour=end_hours, minute=end_minutes)
                except (ValueError, TypeError):
                    end_date = start_date + timedelta(hours=1)
            else:
                end_date = start_date + timedelta(hours=1)
            
            # Format start and end times
            event_body["start"] = {
                "dateTime": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "timeZone": "Asia/Kolkata"
            }
            event_body["end"] = {
                "dateTime": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "timeZone": "Asia/Kolkata"
            }
            
            # Add recurrence if provided
            if ("repeat" in event_data and 
                isinstance(event_data["repeat"], dict) and 
                event_data["repeat"].get("frequency") and 
                isinstance(event_data["repeat"]["frequency"], str) and 
                event_data["repeat"]["frequency"].strip().upper() in ["DAILY", "WEEKLY", "MONTHLY", "YEARLY"]):
                
                repeat_data = event_data["repeat"]
                frequency = repeat_data["frequency"].upper()
                recurrence = [f"RRULE:FREQ={frequency}"]
                
                if repeat_data.get("count") is not None:
                    try:
                        count_val = int(repeat_data["count"])
                        if count_val > 0:
                            recurrence[0] += f";COUNT={count_val}"
                    except (ValueError, TypeError):
                        pass
                elif repeat_data.get("until"):
                    try:
                        until_date = datetime.strptime(repeat_data["until"], "%Y-%m-%d")
                        recurrence[0] += f";UNTIL={until_date.strftime('%Y%m%dT235959Z')}"
                    except ValueError:
                        pass
                
                if len(recurrence[0].split(";")) > 1:
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
        """Execute the event creation"""
        try:
            # Parse input
            event_data = json.loads(query)
            
            # Validate required fields
            if not event_data.get("title"):
                return json.dumps({
                    "success": False,
                    "error": "Event title is required"
                })
            
            if not event_data.get("start_time"):
                return json.dumps({
                    "success": False,
                    "error": "Start time is required for events"
                })
            
            # Create calendar event
            calendar_event = self._create_calendar_event(event_data)
            
            if calendar_event and calendar_event.get("htmlLink"):
                return json.dumps({
                    "success": True,
                    "message": f"Event '{event_data['title']}' created successfully",
                    "event": {
                        "title": event_data["title"],
                        "start_time": event_data.get("start_time"),
                        "end_time": event_data.get("end_time"),
                        "due": event_data.get("due"),
                        "notes": event_data.get("notes"),
                        "calendar_link": calendar_event["htmlLink"]
                    }
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": "Failed to create calendar event"
                })
            
        except json.JSONDecodeError:
            logger.error("Invalid JSON input")
            return json.dumps({
                "success": False,
                "error": "Invalid event data format"
            })
        except Exception as e:
            logger.error(f"Error creating event: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    async def _arun(self, query: str) -> str:
        """Execute the event creation asynchronously"""
        return self._run(query) 