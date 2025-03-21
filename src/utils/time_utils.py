"""Time-related utilities for parsing and formatting dates and times."""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

def parse_date_from_text(content: str) -> str:
    """Parse date from text and return in YYYY-MM-DD format."""
    content_lower = content.lower()
    
    # Remove recurring patterns to avoid confusion
    content_clean = re.sub(r'every\s+\d+\s*(?:day|week|month)s?', '', content_lower)
    
    # Check for relative dates
    if any(day in content_clean for day in ["tomorrow", "tmr"]):
        return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "next week" in content_clean:
        return (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    elif any(day in content_clean for day in ["today", "now"]):
        return datetime.now().strftime("%Y-%m-%d")
    
    # Try to find a specific date
    date_patterns = [
        (r'(\d{2})/(\d{2})(?:/\d{4})?', '%d/%m/%Y'),  # DD/MM or DD/MM/YYYY
        (r'(\d{2})-(\d{2})(?:-\d{4})?', '%d-%m-%Y'),  # DD-MM or DD-MM-YYYY
        (r'\d{4}-\d{2}-\d{2}', '%Y-%m-%d'),           # YYYY-MM-DD
        (r'\d{2}/\d{2}/\d{4}', '%d/%m/%Y')            # DD/MM/YYYY
    ]
    
    for pattern, date_format in date_patterns:
        if matches := re.search(pattern, content):
            try:
                if len(matches.groups()) == 2:  # DD/MM format without year
                    day, month = matches.groups()
                    current_year = datetime.now().year
                    date_str = f"{day}/{month}/{current_year}"
                    parsed_date = datetime.strptime(date_str, '%d/%m/%Y')
                else:
                    date_str = matches.group(0)
                    if len(date_str.split('/')[0]) == 4:  # YYYY/MM/DD format
                        parsed_date = datetime.strptime(date_str, '%Y/%m/%d')
                    else:
                        parsed_date = datetime.strptime(date_str, date_format)
                
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
    
    # Default to today if no date found
    return datetime.now().strftime("%Y-%m-%d")

def parse_time_range(content: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse time range from text and return start and end times in HH:MM format."""
    content_lower = content.lower()
    
    # Time range patterns
    time_patterns = [
        r'(\d{1,2})(?::\d{2})?\s*(?:am|pm)\s*to\s*(\d{1,2})(?::\d{2})?\s*(?:am|pm)',  # 5pm to 6pm
        r'(\d{1,2}):(\d{2})\s*to\s*(\d{1,2}):(\d{2})',  # 17:00 to 18:00
        r'from\s*(\d{1,2})(?::\d{2})?\s*(?:am|pm)\s*to\s*(\d{1,2})(?::\d{2})?\s*(?:am|pm)'  # from 5pm to 6pm
    ]
    
    for pattern in time_patterns:
        if time_match := re.search(pattern, content_lower):
            if len(time_match.groups()) == 2:  # AM/PM format
                start_hour, end_hour = time_match.groups()
                
                # Convert to 24-hour format
                if "pm" in content_lower.split("to")[0].lower():
                    start_hour = str(int(start_hour) + 12) if int(start_hour) < 12 else start_hour
                if "pm" in content_lower.split("to")[1].lower():
                    end_hour = str(int(end_hour) + 12) if int(end_hour) < 12 else end_hour
                
                return f"{int(start_hour):02d}:00", f"{int(end_hour):02d}:00"
            elif len(time_match.groups()) == 4:  # 24-hour format
                start_hour, start_min, end_hour, end_min = time_match.groups()
                return f"{int(start_hour):02d}:{start_min}", f"{int(end_hour):02d}:{end_min}"
    
    # Try to find single time and set duration to 1 hour
    single_time_pattern = r'(?:at\s+)?(\d{1,2})(?::\d{2})?\s*(?:am|pm)'
    if time_match := re.search(single_time_pattern, content_lower):
        hour = time_match.group(1)
        if "pm" in time_match.group(0).lower():
            hour = str(int(hour) + 12) if int(hour) < 12 else hour
        return f"{int(hour):02d}:00", f"{(int(hour) + 1):02d}:00"
    
    return None, None 