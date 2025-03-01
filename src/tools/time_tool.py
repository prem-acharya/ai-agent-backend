import requests
from langchain.tools import BaseTool
from typing import ClassVar

class CurrentTimeTool(BaseTool):
    name: ClassVar[str] = "current_time"
    description: ClassVar[str] = "Fetches the correct current time for a given city using TimeAPI.io."

    def _run(self, city: str = "Kolkata") -> str:
        """Fetch the current time using TimeAPI.io."""
        try:
            timezones = {
                "delhi": "Asia/Kolkata",
                "kolkata": "Asia/Kolkata",
                "new york": "America/New_York",
                "los angeles": "America/Los_Angeles",
                "london": "Europe/London",
                "paris": "Europe/Paris",
                "tokyo": "Asia/Tokyo"
            }

            timezone = timezones.get(city.lower(), "Asia/Kolkata")
            url = f"https://timeapi.io/api/Time/current/zone?timeZone={timezone}"
            response = requests.get(url)
            data = response.json()

            if "dateTime" in data:
                return f"Current time in {city}: {data['dateTime']} (TimeZone: {data['timeZone']})"
            else:
                return f"Error: {data.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, city: str = "Kolkata") -> str:
        """Async version of fetching the current time."""
        return self._run(city)
