from typing import Optional, ClassVar
import logging
import requests
import json
from langchain.tools import BaseTool
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class GetEventsTool(BaseTool):
    """Tool for getting events from Google Calendar"""
    name: ClassVar[str] = "get_events"
    description: ClassVar[str] = """Gets events from Google Calendar.
    Input should be a JSON string with optional fields:
    - today_only: If true, only returns today's events (default: false)
    - tomorrow_only: If true, only returns tomorrow's events (default: false)
    - upcoming_only: If true, only returns future events (default: false)
    - max_results: Maximum number of events to return (default: 10)
    
    Example: {"today_only": true} or {"tomorrow_only": true} or {"upcoming_only": true, "max_results": 5}
    """
    access_token: str
    api_url: str = "https://www.googleapis.com/calendar/v3"
    headers: Optional[dict] = None
    
    def __init__(self, access_token: str):
        super().__init__(access_token=access_token)
        self.api_url = "https://www.googleapis.com/calendar/v3"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    
    def _run(self, query: str) -> str:
        """Execute the events retrieval"""
        try:
            # Parse input
            params = json.loads(query) if query else {}
            today_only = params.get("today_only", False)
            tomorrow_only = params.get("tomorrow_only", False)
            upcoming_only = params.get("upcoming_only", False)
            max_results = params.get("max_results", 10)
            
            # Set time boundaries based on filters
            now = datetime.now()
            time_min = now.isoformat() + 'Z'  # Default to current time
            
            if today_only:
                # Set time_min to start of today
                time_min = datetime(now.year, now.month, now.day, 0, 0, 0).isoformat() + 'Z'
                # Set time_max to end of today
                time_max = datetime(now.year, now.month, now.day, 23, 59, 59).isoformat() + 'Z'
            elif tomorrow_only:
                # Set time bounds for tomorrow
                tomorrow = now + timedelta(days=1)
                time_min = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0).isoformat() + 'Z'
                time_max = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 23, 59, 59).isoformat() + 'Z'
            elif upcoming_only:
                # Already using now as time_min
                time_max = (now + timedelta(days=30)).isoformat() + 'Z'  # Next 30 days
            else:
                # Default: get all events from now
                time_max = (now + timedelta(days=30)).isoformat() + 'Z'  # Next 30 days
            
            # Build request URL with parameters
            calendar_request_url = f"{self.api_url}/calendars/primary/events"
            calendar_params = {
                "maxResults": max_results,
                "timeMin": time_min,
                "singleEvents": "true",
                "orderBy": "startTime"
            }
            
            if 'time_max' in locals():
                calendar_params["timeMax"] = time_max
            
            # Get events from primary calendar
            events_response = requests.get(
                calendar_request_url,
                headers=self.headers,
                params=calendar_params
            )
            events_response.raise_for_status()
            events_data = events_response.json()
            events = events_data.get("items", [])
            
            # Format events for display
            formatted_events = []
            for event in events:
                # Extract start and end time
                start = event.get("start", {})
                end = event.get("end", {})
                
                # Determine if it's an all-day event
                is_all_day = "date" in start and "date" in end
                
                formatted_event = {
                    "title": event.get("summary", "Untitled Event"),
                    "description": event.get("description", ""),
                    "location": event.get("location", ""),
                    "is_all_day": is_all_day,
                    "start": start.get("dateTime" if not is_all_day else "date", ""),
                    "end": end.get("dateTime" if not is_all_day else "date", ""),
                    "link": event.get("htmlLink", ""),
                    "meet_link": event.get("hangoutLink", ""),
                    "attendees": [
                        {
                            "email": attendee.get("email", ""),
                            "name": attendee.get("displayName", ""),
                            "status": attendee.get("responseStatus", "")
                        }
                        for attendee in event.get("attendees", [])
                    ]
                }
                
                formatted_events.append(formatted_event)
            
            return json.dumps({
                "success": True,
                "events": formatted_events
            })
            
        except Exception as e:
            logger.error(f"Error getting events: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    async def _arun(self, query: str) -> str:
        """Execute the events retrieval asynchronously"""
        return self._run(query) 