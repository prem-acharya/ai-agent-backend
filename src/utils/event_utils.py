import re
import json
import logging
from typing import Dict, Any
from datetime import datetime, timedelta
from langchain.schema import HumanMessage

from src.utils.time_utils import parse_date_from_text, parse_time_range
from src.utils.prompt.event_prompts import get_event_analysis_prompt

logger = logging.getLogger(__name__)

def format_event_details(event_data: Dict[str, Any]) -> str:
    """Format event details for display to the user."""
    details = []
    
    # Add title with emoji
    summary = event_data.get('summary', 'New Event')
    if not summary.startswith('📅'):
        summary = f"📅 {summary}"
    details.append(f"**{summary}**\n")
    
    # Add date and time
    date_str = event_data.get("due", "")
    start_time = event_data.get("start_time", "")
    end_time = event_data.get("end_time", "")
    
    if date_str and start_time and end_time:
        details.append(f"⏰ **When:** {date_str} from {start_time} to {end_time}")
    elif date_str and start_time:
        details.append(f"⏰ **When:** {date_str} at {start_time}")
    elif date_str:
        details.append(f"⏰ **When:** {date_str}")
        
    # Add location
    location = event_data.get("location", "")
    if location:
        if location.lower() == "google meet":
            details.append(f"📍 **Where:** Virtual Meeting (Google Meet)")
        else:
            details.append(f"📍 **Where:** {location}")
        
    # Add description
    description = event_data.get("description", "")
    if description:
        details.append(f"📝 **Description:** {description}")
        
    # Add attendees
    attendees = event_data.get("attendees", [])
    if attendees:
        if isinstance(attendees, list):
            if all(isinstance(item, dict) for item in attendees):
                emails = [attendee.get("email", "") for attendee in attendees if attendee.get("email")]
                if emails:
                    details.append(f"👥 **Attendees:** {', '.join(emails)}")
            else:
                details.append(f"👥 **Attendees:** {', '.join(attendees)}")
        elif isinstance(attendees, str):
            details.append(f"👥 **Attendees:** {attendees}")
            
    # Add recurrence
    recurrence = event_data.get("recurrence", [])
    if recurrence:
        if isinstance(recurrence, list) and len(recurrence) > 0:
            recurrence_text = recurrence[0].replace("RRULE:", "")
            parts = recurrence_text.split(";")
            
            # Simplify the recurrence rule for display
            freq = next((part.replace("FREQ=", "") for part in parts if part.startswith("FREQ=")), "")
            count = next((part.replace("COUNT=", "") for part in parts if part.startswith("COUNT=")), "")
            byday = next((part.replace("BYDAY=", "") for part in parts if part.startswith("BYDAY=")), "")
            
            recurrence_str = f"🔄 **Repeats:** {freq.lower()}"
            if count:
                recurrence_str += f" for {count} times"
            if byday:
                days = byday.split(",")
                day_names = {
                    "MO": "Monday", "TU": "Tuesday", "WE": "Wednesday", 
                    "TH": "Thursday", "FR": "Friday", "SA": "Saturday", "SU": "Sunday"
                }
                day_str = ", ".join(day_names.get(day, day) for day in days)
                recurrence_str += f" on {day_str}"
                
            details.append(recurrence_str)
    
    # Add reminders
    reminders = event_data.get("reminders", [])
    if reminders:
        reminder_strs = []
        for reminder in reminders:
            minutes = reminder.get("minutes", 0)
            method = reminder.get("method", "notification")
            if minutes >= 1440:  # 24 hours or more
                days = minutes // 1440
                reminder_strs.append(f"{days} day(s) before by {method}")
            elif minutes >= 60:  # 1 hour or more
                hours = minutes // 60
                reminder_strs.append(f"{hours} hour(s) before by {method}")
            else:
                reminder_strs.append(f"{minutes} minutes before by {method}")
        if reminder_strs:
            details.append(f"⏰ **Reminders:** {', '.join(reminder_strs)}")
            
    # Format as a single string with proper spacing
    return "\n".join(details)

async def prepare_event_data(content: str, llm) -> Dict[str, Any]:
    """Prepare event data from user input using AI analysis."""
    try:
        # Get AI analysis of the event
        analysis_prompt = get_event_analysis_prompt().format(content=content)
        response = await llm.agenerate([[HumanMessage(content=analysis_prompt)]])
        response_text = response.generations[0][0].text.strip()
        logger.info(f"AI Response: {response_text}")
        
        # Try to extract JSON from the response
        try:
            # Clean up the response text to ensure valid JSON
            # First, try to find JSON in code blocks
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response_text)
            if not json_match:
                # If not in code blocks, try to find bare JSON
                json_match = re.search(r'\{[\s\S]*?\}', response_text)
            
            if json_match:
                event_analysis = json.loads(json_match.group(1) if '```' in response_text else json_match.group(0))
                logger.info(f"AI Analysis: {json.dumps(event_analysis, indent=2)}")
            else:
                logger.error("No JSON found in response")
                # Try to get a new analysis with a more specific prompt
                analysis_prompt = f"""Analyze this event request and generate a concise JSON response:
                {content}
                
                Focus on creating a brief, relevant description that captures the main purpose."""
                
                response = await llm.agenerate([[HumanMessage(content=analysis_prompt)]])
                response_text = response.generations[0][0].text.strip()
                json_match = re.search(r'\{[\s\S]*?\}', response_text)
                if json_match:
                    event_analysis = json.loads(json_match.group(0))
                else:
                    event_analysis = {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            event_analysis = {}
        
        # Extract basic event data
        event_date = parse_date_from_text(content)
        start_time, end_time = parse_time_range(content)
        
        # Extract all email addresses from the content
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        found_emails = re.findall(email_pattern, content)
        
        # Build the event data with AI-generated content or fallback to generated content
        event_data = {}
        
        # If we have AI analysis, use it as the base
        if event_analysis:
            event_data = event_analysis.copy()
            
            # Override or set specific fields
            if start_time:
                event_data["start_time"] = start_time
            if end_time:
                event_data["end_time"] = end_time
            if event_date:
                event_data["due"] = event_date
            
            # Always ensure attendees are included
            attendees_list = []
            # First, add any attendees from the AI analysis
            if event_analysis.get("attendees"):
                if isinstance(event_analysis["attendees"], list):
                    for attendee in event_analysis["attendees"]:
                        if isinstance(attendee, dict) and "email" in attendee:
                            attendee_data = {
                                "email": attendee["email"],
                                "responseStatus": "needsAction",
                                "optional": False
                            }
                            attendees_list.append(attendee_data)
                        elif isinstance(attendee, str):
                            attendee_data = {
                                "email": attendee,
                                "responseStatus": "needsAction",
                                "optional": False
                            }
                            attendees_list.append(attendee_data)
            
            # Then add any emails found in the content that aren't already included
            existing_emails = {attendee["email"] for attendee in attendees_list}
            for email in found_emails:
                if email not in existing_emails:
                    attendee_data = {
                        "email": email.strip(),
                        "responseStatus": "needsAction",
                        "optional": False
                    }
                    attendees_list.append(attendee_data)
            
            event_data["attendees"] = attendees_list
            
            # Ensure we have a description
            if not event_data.get("description"):
                # Request a specific description from the AI
                desc_prompt = f"""Generate a brief, focused description for this event:
                {content}
                
                Keep it concise with:
                1. One to three lines overview
                2. 2-3 key points maximum (Agenda with specific points)"""
                
                desc_response = await llm.agenerate([[HumanMessage(content=desc_prompt)]])
                desc_text = desc_response.generations[0][0].text.strip()
                event_data["description"] = desc_text
            
            # Ensure conference data is set
            event_data["create_conference"] = event_analysis.get("create_conference", True)
            
            # Add or update location if not present
            if not event_data.get("location"):
                event_data["location"] = "Google Meet"
                
            # Add or update reminders if not present
            if not event_data.get("reminders"):
                event_data["reminders"] = [
                    {"method": "email", "minutes": 1440},
                    {"method": "email", "minutes": 60},
                    {"method": "popup", "minutes": 10}
                ]
                
            # Add recurrence if specified in content but not in analysis
            if ("weekly" in content.lower() or "every week" in content.lower()) and not event_data.get("recurrence"):
                event_data["recurrence"] = ["RRULE:FREQ=WEEKLY"]
        else:
            # Extract key terms for the summary
            meeting_type = "Meeting"
            emoji = "📅"  # Default emoji
            
            if "team" in content.lower():
                meeting_type = "Team Sync"
                emoji = "🤝"
            elif "review" in content.lower():
                meeting_type = "Review"
                emoji = "📋"
            
            # Create a concise description
            description = f"<b>{emoji} {meeting_type}</b><br><br>"
            description += f"<i>Brief sync to discuss key points and updates.</i><br><br>"
            description += "<b>🎯 Key Points</b><br>"
            description += "• Updates and progress<br>"
            description += "• Discussion items<br>"
            
            # Fallback to basic event data
            event_data = {
                "summary": f"{emoji} {meeting_type}",
                "description": description,
                "location": "Google Meet",
                "start_time": start_time or "10:00",
                "end_time": end_time or "11:00",
                "due": event_date,
                "create_conference": True,
                "attendees": [{"email": email.strip(), "responseStatus": "needsAction", "optional": False} for email in found_emails],
                "recurrence": ["RRULE:FREQ=WEEKLY"] if "weekly" in content.lower() or "every week" in content.lower() else [],
                "reminders": [
                    {"method": "email", "minutes": 1440},
                    {"method": "email", "minutes": 60},
                    {"method": "popup", "minutes": 10}
                ]
            }
        
        logger.info(f"Prepared event data: {json.dumps(event_data, indent=2)}")
        return event_data
        
    except Exception as e:
        logger.error(f"Error getting event analysis from Gemini: {str(e)}")
        # Fallback to basic event data
        return {
            "summary": "📅 New Meeting",
            "description": "<b>📅 Meeting</b><br><br><i>Brief sync to discuss updates.</i><br><br><b>🎯 Key Points</b><br>• Updates<br>• Discussion<br>• Next steps",
            "location": "Google Meet",
            "start_time": start_time or "10:00",
            "end_time": end_time or "11:00",
            "due": event_date,
            "create_conference": True,
            "attendees": [{"email": email.strip(), "responseStatus": "needsAction", "optional": False} for email in found_emails],
            "recurrence": ["RRULE:FREQ=WEEKLY"] if "weekly" in content.lower() or "every week" in content.lower() else [],
            "reminders": [
                {"method": "email", "minutes": 1440},
                {"method": "email", "minutes": 60},
                {"method": "popup", "minutes": 10}
            ]
        } 