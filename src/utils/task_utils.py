"""Task-related utilities for preparing task and event data."""

import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from .time_utils import parse_date_from_text, parse_time_range

def clean_title(title: str) -> str:
    """Clean task/event title by removing time information and date terms."""
    # Remove time information
    time_patterns = [
        r'\s*at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)',  # at 2 PM
        r'\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)',       # 2 PM
        r'\s*\d{2}:\d{2}'                           # 14:00
    ]
    for pattern in time_patterns:
        title = re.sub(pattern, '', title, flags=re.IGNORECASE)
    
    # Remove date terms
    date_terms = ["today", "tomorrow", "next week", "next month", "next day"]
    for term in date_terms:
        title = re.sub(r'\b' + term + r'\b', '', title, flags=re.IGNORECASE)
    
    # Cleanup any double spaces created by removals
    return re.sub(r'\s+', ' ', title).strip()

def extract_task_title(content: str) -> Optional[str]:
    """Extract task title from user input."""
    content_lower = content.lower()
    
    # Try task markers first
    task_markers = ["remind me to ", "set reminder to ", "create task to ", "set task to "]
    for marker in task_markers:
        if marker in content_lower:
            return content_lower.split(marker, 1)[1].strip()
    
    # Fallback extraction
    if "to " in content_lower:
        return content_lower.split("to ", 1)[1].strip()
    
    # Last resort
    return content_lower.replace("reminder", "").replace("task", "").replace("set", "").replace("create", "").strip()

def extract_event_title(content: str) -> Optional[str]:
    """Extract event title from user input."""
    content_lower = content.lower()
    
    # Try event markers first
    event_markers = ["schedule meeting for ", "create meeting for ", "set meeting for ",
                    "schedule event for ", "create event for ", "set event for "]
    for marker in event_markers:
        if marker in content_lower:
            return content_lower.split(marker, 1)[1].strip()
    
    # Fallback extraction
    if "meeting about " in content_lower:
        return content_lower.split("meeting about ", 1)[1].strip()
    elif "event about " in content_lower:
        return content_lower.split("event about ", 1)[1].strip()
    
    # Last resort
    return content_lower.replace("meeting", "").replace("event", "").replace("schedule", "").replace("create", "").replace("set", "").strip()

def extract_attendees(content: str) -> Optional[List[str]]:
    """Extract attendees from content."""
    content_lower = content.lower()
    
    if "guest list:" in content_lower:
        attendees = content.lower().split("guest list:")[1].strip().split(",")
        return [email.strip() for email in attendees]
    elif "email is" in content_lower:
        email = content.lower().split("email is")[1].strip()
        return [email]
    
    return None

def extract_notes(content: str) -> Optional[str]:
    """Extract notes from content."""
    if "notes:" in content.lower():
        return content.split("notes:", 1)[1].strip()
    return None

def format_task_details(task_data: Dict[str, Any]) -> str:
    """Format task details for display."""
    details = []
    
    # Title with emoji
    if task_data.get("title"):
        title = task_data.get("title", "")
        # If repeat settings exist, add them to the title display
        if task_data.get("repeat") and "🔄" not in title:
            repeat = task_data["repeat"]
            count = repeat.get("count", 1)
            details.append(f"📝 **Task**: {title} 🔄 Repeats {count} times")
        else:
            details.append(f"📝 **Task**: {title}")
    
    # Due date
    if task_data.get("due"):
        details.append(f"📅 **Due**: {task_data['due']}")
    
    # Time
    if task_data.get("time"):
        details.append(f"⏰ **Time**: {task_data['time']}")
    
    # Repeat settings
    if task_data.get("repeat"):
        repeat = task_data["repeat"]
        count = repeat.get("count", 1)
        details.append(f"🔄 **Repeats**: {count} times")
    
    # Category and Priority
    if task_data.get("category"):
        category_emojis = {
            "health": "🏥",
            "work": "💼",
            "personal": "👤",
            "fitness": "💪",
            "study": "📚",
            "shopping": "🛒",
            "home": "🏠"
        }
        emoji = category_emojis.get(task_data["category"].lower(), "📌")
        details.append(f"{emoji} **Category**: {task_data['category'].title()}")
    
    if task_data.get("priority"):
        priority_emojis = {
            "high": "🔴",
            "medium": "🟡",
            "low": "🟢"
        }
        emoji = priority_emojis.get(task_data["priority"].lower(), "📌")
        details.append(f"{emoji} **Priority**: {task_data['priority'].title()}")
    
    # Description
    if task_data.get("description"):
        details.append(f"\n📋 **Description**:\n{task_data['description']}")
    
    # Notes
    if task_data.get("notes"):
        details.append(f"\n📝 **Notes**:\n{task_data['notes']}")
    
    # Estimated time
    if task_data.get("estimated_time"):
        try:
            minutes = int(task_data["estimated_time"])
            if minutes < 60:
                time_str = f"{minutes} minute{'s' if minutes != 1 else ''}"
            else:
                hours = minutes // 60
                remaining_mins = minutes % 60
                time_str = f"{hours} hour{'s' if hours != 1 else ''}"
                if remaining_mins:
                    time_str += f" {remaining_mins} minute{'s' if remaining_mins != 1 else ''}"
            details.append(f"⏱️ **Estimated Time**: {time_str}")
        except (ValueError, TypeError):
            pass
    
    return "\n\n".join(details)

def prepare_task_data(content: str, task_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Prepare task data from user input and optional AI analysis."""
    task_data = {}
    
    # Get title from AI analysis or extract from content
    if task_analysis and task_analysis.get("title"):
        task_data["title"] = task_analysis["title"]
    else:
        title = extract_task_title(content)
        task_data["title"] = clean_title(title) if title else "New Task"
    
    # Get due date
    task_data["due"] = parse_date_from_text(content)
    
    # Get time (default 10:00)
    task_data["time"] = task_analysis.get("time", "10:00") if task_analysis else "10:00"
    
    # Get repeat settings
    content_lower = content.lower()
    if "every" in content_lower or "repeat" in content_lower or "recurring" in content_lower:
        repeat_data = {"count": 1}  # Default to 1 times
        if task_analysis and task_analysis.get("repeat"):
            repeat_data = task_analysis["repeat"]
        else:
            # Look for count in the text
            count_patterns = [
                r'(?:for|count|repeat(?:s|ing)?)\s+(\d+)(?:\s+times)?',
                r'(\d+)\s+times',
                r'(\d+)\s+(?:occurrence|iteration)s?'
            ]
            
            for pattern in count_patterns:
                count_match = re.search(pattern, content_lower)
                if count_match:
                    repeat_data["count"] = int(count_match.group(1))
                    break
        
        task_data["repeat"] = repeat_data
    
    # Prepare notes and description
    if task_analysis:
        if task_analysis.get("description"):
            task_data["description"] = task_analysis["description"]
        
        if task_analysis.get("notes"):
            task_data["notes"] = task_analysis["notes"] if isinstance(task_analysis["notes"], str) else "\n".join(task_analysis["notes"])
    else:
        # Fallback to basic notes if no AI analysis
        notes = []
        
        # Add repeat information to notes if present
        if task_data.get("repeat"):
            repeat = task_data["repeat"]
            notes.append(f"🔄 Repeats every {repeat['interval']} {repeat['unit']}")
        
        # Add time information
        notes.append(f"⏰ Set for {task_data['time']}")
        
        user_notes = extract_notes(content)
        if user_notes:
            notes.append(f"👤 {user_notes}")
        
        if len(notes) < 3:
            notes.extend([
                "📌 Remember to stay consistent",
                "💡 Track your progress"
            ])
        
        task_data["notes"] = "\n".join(notes[:3])  # Limit to 3 points
    
    return task_data

def prepare_event_data(content: str) -> Dict[str, Any]:
    """Prepare event data from user input."""
    event_data = {}
    
    # Get title
    title = extract_event_title(content)
    event_data["title"] = clean_title(title).title() if title else "New Event"
    
    # Get date
    event_data["due"] = parse_date_from_text(content)
    
    # Get time range
    start_time, end_time = parse_time_range(content)
    if start_time:
        event_data["start_time"] = start_time
    if end_time:
        event_data["end_time"] = end_time
    
    # Get attendees
    attendees = extract_attendees(content)
    if attendees:
        event_data["attendees"] = attendees
    
    # Get notes
    notes = extract_notes(content)
    if notes:
        event_data["notes"] = notes
    
    return event_data 