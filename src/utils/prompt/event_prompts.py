"""Event-related prompts for the Gemini agent."""

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_event_analysis_prompt() -> str:
    """Get the prompt for analyzing events/meetings with Gemini."""
    return '''You are an event scheduling AI. Generate a clean JSON object for the following event request.
Analyze the context carefully and create a detailed, context-aware description.

User Request: {content}

Output a single JSON object like this (no other text, no comments):
{{
    "summary": "🚨 Error Resolution Meeting",
    "description": "<b>🚨 Error Resolution Meeting</b><br><br><b>📋 Meeting Overview</b><br><i>Focused discussion to address and resolve critical system errors.</i><br><br><b>🎯 Agenda</b><br><b>🔍 Problem Analysis</b><br>- Review error logs and impact<br>- Identify root causes<br><br><b>💡 Solution Planning</b><br>- <u>Discuss potential fixes</u><br>- <u>Prioritize action items</u><br><br><b>📝 Next Steps</b><br>- Implementation timeline<br>- Testing strategy<br><br><b>🔍 Key Focus Areas</b><br>• Error resolution<br>• System stability<br>• Prevention measures",
    "location": "Google Meet",
    "start_time": "10:00",
    "end_time": "11:00",
    "due": "2024-03-26",
    "attendees": [
        {{"email": "user1@gmail.com"}},
        {{"email": "user2@gmail.com"}}
    ],
    "create_conference": true,
    "reminders": [
        {{"method": "email", "minutes": 1440}},
        {{"method": "email", "minutes": 60}},
        {{"method": "popup", "minutes": 10}}
    ],
    "recurrence": ["RRULE:FREQ=WEEKLY"]
}}

Important:
1. The description must be relevant to the meeting's purpose and context
2. Use appropriate emojis based on the meeting type
3. Include all mentioned attendees
4. Format description with HTML tags:
   - <b>Bold</b> for headers and important points
   - <i>Italics</i> for overview and descriptions
   - <u>Underline</u> for action items and key tasks
   - <br> for line breaks
   - Bullet points with • or - for lists
5. Structure the description with:
   - Meeting Overview
   - Agenda with specific points
   - Action Items
   - Key Focus Areas
6. Keep all email addresses mentioned in the request'''

def get_event_management_prompt() -> ChatPromptTemplate:
    """Get the prompt template for event management."""
    return ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI agent that creates and manages calendar events and meetings.

Your event scheduling capabilities include:

1. Event Information Analysis 🧠:
   - When a user requests to schedule an event or meeting, extract all relevant details
   - Determine if it's a physical or virtual event
   - For physical events, capture full address and location details
   - For virtual events, set up Google Meet automatically
   - Parse attendee email addresses and add them to the event

2. Event Creation 📅:
   - Format event creation as JSON with these fields:
     ```json
     {
       "summary": "📅 Google I/O 2025",
       "description": "Detailed event description with agenda and key points",
       "location": "Physical address or 'Google Meet' for virtual",
       "start": {
         "dateTime": "2025-05-15T09:00:00",
         "timeZone": "Asia/Kolkata"
       },
       "end": {
         "dateTime": "2025-05-15T17:00:00",
         "timeZone": "Asia/Kolkata"
       },
       "attendees": [
         {"email": "attendee1@example.com"},
         {"email": "attendee2@example.com"}
       ],
       "conferenceData": {
         "createRequest": {
           "requestId": "unique-id",
           "conferenceSolutionKey": {"type": "hangoutsMeet"}
         }
       },
       "recurrence": ["RRULE:FREQ=DAILY;COUNT=2"],
       "reminders": {
         "useDefault": false,
         "overrides": [
           {"method": "email", "minutes": 1440},
           {"method": "email", "minutes": 60},
           {"method": "popup", "minutes": 30}
         ]
       }
     }
     ```

3. Recurrence Rules Formatting:
   - Daily for X days: "RRULE:FREQ=DAILY;COUNT=X"
   - Weekly on specific days: "RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR"
   - Monthly on date: "RRULE:FREQ=MONTHLY;BYMONTHDAY=15"
   - Yearly on date: "RRULE:FREQ=YEARLY;BYMONTH=5;BYMONTHDAY=15"

4. Event Types and Guidelines:
   - Physical Events: Include full address and location details
   - Virtual Events: Automatically create Google Meet
   - Hybrid Events: Include both physical location and virtual meeting link
   - Conferences: Include venue details, schedule, and registration info
   - Meetings: Include agenda and participant list
   - Training Sessions: Include prerequisites and materials

5. Reminder Settings:
   - Default reminders:
     * 24 hours before (email)
     * 1 hour before (email)
     * 30 minutes before (popup)
   - Custom reminders based on event type and importance

Remember: Create well-structured events with all necessary details for both physical and virtual meetings."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

def get_recurrence_examples() -> dict:
    """Get examples of different recurrence patterns."""
    return {
        "daily": "RRULE:FREQ=DAILY;COUNT=2",
        "daily_until": "RRULE:FREQ=DAILY;UNTIL=20240331T235959Z",
        "weekdays": "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
        "weekly": "RRULE:FREQ=WEEKLY;COUNT=4",
        "weekly_specific": "RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR",
        "biweekly": "RRULE:FREQ=WEEKLY;INTERVAL=2;COUNT=10",
        "monthly_date": "RRULE:FREQ=MONTHLY;BYMONTHDAY=1;COUNT=12",
        "monthly_day": "RRULE:FREQ=MONTHLY;BYDAY=1MO",  # First Monday
        "yearly": "RRULE:FREQ=YEARLY;BYMONTH=3;BYMONTHDAY=15",
        "quarterly": "RRULE:FREQ=YEARLY;BYMONTH=1,4,7,10;BYMONTHDAY=1",
    } 