
from langchain.tools import BaseTool
from src.services.weather_service import WeatherService
import json
from typing import ClassVar

class WeatherTool(BaseTool):
    name: ClassVar[str] = "weather"
    description: ClassVar[str] = "Get weather information for a city. Input should be a city name."
    weather_service: ClassVar[WeatherService] = WeatherService()

    async def _arun(self, city: str) -> str:
        """Get weather information for a city."""
        try:
            weather_data = await self.weather_service.get_weather_by_city(city.strip())
            
            # Format weather data nicely
            formatted_data = {
                "location": weather_data["name"],
                "weather": weather_data["weather"][0]["description"],
                "temperature": {
                    "current": weather_data["main"]["temp"],
                    "feels_like": weather_data["main"]["feels_like"]
                },
                "humidity": weather_data["main"]["humidity"],
                "wind": {
                    "speed": weather_data["wind"]["speed"],
                    "direction": weather_data["wind"]["deg"]
                },
                "visibility": weather_data.get("visibility", "N/A")
            }
            
            return json.dumps(formatted_data, indent=2)
        except Exception as e:
            return f"Error getting weather data for {city}: {str(e)}"

    def _run(self, city: str) -> str:
        raise NotImplementedError("WeatherTool only supports async operations")
