from typing import Optional, ClassVar, Dict
import logging
import requests
import json
import uuid
from langchain.tools import BaseTool
from datetime import datetime, timedelta
from pydantic import Field

logger = logging.getLogger(__name__)

class CreateEventTool(BaseTool):
    """Tool for creating events/meetings in Google Calendar"""
    name: ClassVar[str] = "create_event"
    description: ClassVar[str] = """Creates a new event/meeting in Google Calendar.
    Input should be a JSON string with required fields:
    - summary: The event title (required)
    - description: Event description/details (optional)
    - location: Physical location or "Google Meet" (optional)
    - start_time: Start time in 24-hour format "HH:MM" (required)
    - end_time: End time in 24-hour format "HH:MM" (required)
    - due: Event date (required, can be "today", "tomorrow", or "YYYY-MM-DD")
    - notes: Additional notes/description (optional)
    - attendees: List of attendee email addresses (optional)
    - create_conference: Boolean to create Google Meet (optional, default true for virtual meetings)
    - reminders: List of reminders with method and minutes (optional)
    - repeat: Recurrence settings (optional) with fields:
      - frequency: "daily", "weekly", "monthly", or "yearly"
      - count: Number of occurrences (optional)
      - until: End date in "YYYY-MM-DD" format (optional)
      - byday: Days of week like "MO,TU,WE,TH,FR" (optional)
      - interval: Interval between occurrences (optional)
    
    Example: {
        "summary": "Team Meeting",
        "description": "Discuss project progress",
        "location": "Google Meet",
        "start_time": "14:30",
        "end_time": "15:30",
        "due": "2024-03-19",
        "attendees": ["user@example.com"],
        "create_conference": true,
        "reminders": [
            {"method": "email", "minutes": 30},
            {"method": "popup", "minutes": 10}
        ],
        "repeat": {
            "frequency": "do not repeat",
            "count": 4,
            "byday": "MO"
        }
    }
    """
    access_token: str = Field(description="Google Calendar API access token")
    calendar_api_url: str = Field(default="https://www.googleapis.com/calendar/v3", description="Google Calendar API URL")
    headers: Dict[str, str] = Field(default_factory=dict, description="Request headers")
    
    def __init__(self, access_token: str):
        """Initialize the tool with access token"""
        super().__init__(
            access_token=access_token,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
        )

    def _create_calendar_event(self, event_data: dict) -> dict:
        """Create a Calendar event"""
        try:
            # Format event data with clean summary
            summary = event_data.get("summary", event_data.get("title", "New Event"))
            
            event_body = {
                "summary": summary,
                "description": event_data.get("description", event_data.get("notes", "")),
                "location": event_data.get("location", "Google Meet")
            }
            
            # Get user info to set as organizer
            try:
                user_info_response = requests.get(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers=self.headers
                )
                user_info_response.raise_for_status()
                user_info = user_info_response.json()
                
                # Set creator and organizer
                event_body["creator"] = {
                    "email": user_info.get("email"),
                    "self": True
                }
                event_body["organizer"] = event_body["creator"]
                
            except Exception as e:
                logger.error(f"Error getting user info: {str(e)}")
            
            # Add attendees if provided
            if "attendees" in event_data and event_data["attendees"]:
                if isinstance(event_data["attendees"], list):
                    event_body["attendees"] = []
                    for attendee in event_data["attendees"]:
                        if isinstance(attendee, dict) and "email" in attendee:
                            event_body["attendees"].append({
                                "email": attendee["email"],
                                "responseStatus": "needsAction"
                            })
                        elif isinstance(attendee, str):
                            event_body["attendees"].append({
                                "email": attendee.strip(),
                                "responseStatus": "needsAction"
                            })
            
            # Set start and end times
            date_str = event_data.get("due", "today")
            start_time = event_data.get("start_time", "09:00")
            end_time = event_data.get("end_time")
            
            # Parse date
            if date_str.lower() == "today":
                start_date = datetime.now()
            elif date_str.lower() == "tomorrow":
                start_date = datetime.now() + timedelta(days=1)
            else:
                try:
                    start_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    start_date = datetime.now()
            
            # Ensure start_date is not in the past
            now = datetime.now()
            if start_date.date() < now.date():
                start_date = now
            
            # Parse start time
            try:
                hours, minutes = map(int, start_time.split(":"))
                start_date = start_date.replace(hour=hours, minute=minutes, second=0, microsecond=0)
            except (ValueError, TypeError):
                start_date = start_date.replace(hour=9, minute=0, second=0, microsecond=0)
            
            # Set end time
            if end_time:
                try:
                    end_hours, end_minutes = map(int, end_time.split(":"))
                    end_date = start_date.replace(hour=end_hours, minute=end_minutes)
                    if end_date < start_date:
                        end_date = end_date + timedelta(days=1)
                except (ValueError, TypeError):
                    end_date = start_date + timedelta(hours=1)
            else:
                end_date = start_date + timedelta(hours=1)
            
            # Format datetime strings in ISO format with timezone
            event_body["start"] = {
                "dateTime": start_date.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
                "timeZone": "Asia/Kolkata"
            }
            
            event_body["end"] = {
                "dateTime": end_date.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
                "timeZone": "Asia/Kolkata"
            }
            
            # Add conferenceData if requested
            create_conference = event_data.get("create_conference", True)
            if create_conference and event_data.get("location", "").lower() in ["google meet", "virtual meeting", "online"]:
                request_id = str(uuid.uuid4())
                event_body["conferenceData"] = {
                    "createRequest": {
                        "requestId": request_id,
                        "conferenceSolutionKey": {"type": "hangoutsMeet"}
                    }
                }
            
            # Add reminders
            event_body["reminders"] = {
                "useDefault": False,
                "overrides": [
                    {"method": "email", "minutes": 1440},
                    {"method": "email", "minutes": 60},
                    {"method": "popup", "minutes": 10}
                ]
            }
            
            # Create event
            response = requests.post(
                f"{self.calendar_api_url}/calendars/primary/events?conferenceDataVersion=1&sendUpdates=all",
                headers=self.headers,
                json=event_body
            )
            response.raise_for_status()
            created_event = response.json()
            
            return {
                "id": created_event.get("id", ""),
                "htmlLink": created_event.get("htmlLink", ""),
                "hangoutLink": created_event.get("hangoutLink", "") if create_conference else ""
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Calendar API request error: {str(e)}")
            logger.error(f"Request body: {json.dumps(event_body, indent=2)}")
            raise
        except Exception as e:
            logger.error(f"Error creating calendar event: {str(e)}")
            raise

    def _run(self, query: str) -> str:
        """Execute the event creation"""
        try:
            # Parse input
            event_data = json.loads(query)
            
            # Validate required fields
            if not event_data.get("summary") and not event_data.get("title"):
                return json.dumps({
                    "success": False,
                    "error": "Event title/summary is required"
                })
            
            # Create calendar event
            calendar_event = self._create_calendar_event(event_data)
            
            if calendar_event and calendar_event.get("htmlLink"):
                # Prepare response with all the details used to create the event
                title = event_data.get("summary", event_data.get("title", "New Event"))
                response_data = {
                    "success": True,
                    "message": f"Event '{title}' created successfully",
                    "event": {
                        "title": title,
                        "description": event_data.get("description", event_data.get("notes", "")),
                        "location": event_data.get("location", ""),
                        "start_time": event_data.get("start_time", ""),
                        "end_time": event_data.get("end_time", ""),
                        "due": event_data.get("due", ""),
                        "attendees": event_data.get("attendees", []),
                        "calendar_link": calendar_event["htmlLink"]
                    }
                }
                
                # Add hangout link if available
                if calendar_event.get("hangoutLink"):
                    response_data["event"]["hangout_link"] = calendar_event["hangoutLink"]
                
                # Add recurrence info if available
                if "recurrence" in event_data or "repeat" in event_data:
                    response_data["event"]["recurrence"] = event_data.get("recurrence", None) or event_data.get("repeat", None)
                
                # Add reminders if available
                if "reminders" in event_data:
                    response_data["event"]["reminders"] = event_data["reminders"]
                
                return json.dumps(response_data)
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