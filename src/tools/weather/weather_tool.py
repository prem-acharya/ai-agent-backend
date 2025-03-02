import json
from langchain.tools import BaseTool
from typing import ClassVar, Dict
import httpx
import os

class WeatherService:

    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")
        if not self.api_key:
            raise ValueError("WEATHER_API_KEY not found in environment variables")

        self.base_url = os.getenv("WEATHER_API_URL")

    async def get_weather_by_city(self, city: str, units: str = "metric", lang: str = "en") -> Dict:
        try:
            # Clean the city input
            clean_city = city.strip().replace('\n', '').replace('\r', '')
            
            params = {
                "q": clean_city,
                "appid": self.api_key,
                "units": units,
                "lang": lang
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(self.base_url, params=params, timeout=10.0)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            error_msg = f"HTTP error occurred: {str(e)}"
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error fetching weather data: {str(e)}"
            raise ValueError(error_msg)


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
