from typing import Dict, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from .weather_service import WeatherService
import logging

logger = logging.getLogger(__name__)

class WeatherInput(BaseModel):
    city: str
    units: str = "metric"
    ai_response: str = None

class WeatherTool(BaseTool):
    name: str = "weather"
    description: str = """
    Get current weather information for a specific city.
    Input should be a dictionary with 'city' key and optionally 'units' (metric/imperial) and 'ai_response'.
    Example: {"city": "London", "units": "metric", "ai_response": "Your AI model response here."}
    """
    args_schema: type[BaseModel] = WeatherInput
    weather_service: WeatherService = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.weather_service is None:
            self.weather_service = WeatherService()

    async def _arun(self, tool_input: Dict) -> str:
        try:
            city = tool_input["city"]
            units = tool_input.get("units", "metric")
            ai_response = tool_input.get("ai_response", "No AI response provided.")
            
            # Add AI thinking message
            ai_message = "I will call the weather agent to get the current weather data."

            weather_data = await self.weather_service.get_weather_by_city(
                city=city,
                units=units
            )
            
            temp = weather_data.get("main", {}).get("temp")
            feels_like = weather_data.get("main", {}).get("feels_like")
            humidity = weather_data.get("main", {}).get("humidity")
            weather_desc = weather_data.get("weather", [{}])[0].get("description", "")
            
            return (
                f"{ai_message}\n"
                f"Current weather in {city}:\n"
                f"Temperature: {temp}°{'C' if units == 'metric' else 'F'}\n"
                f"Feels like: {feels_like}°{'C' if units == 'metric' else 'F'}\n"
                f"Humidity: {humidity}%\n"
                f"Conditions: {weather_desc.capitalize()}\n"
            )
            
        except Exception as e:
            logger.error(f"Weather tool error: {str(e)}")
            return f"Error getting weather data: {str(e)}"

    def _run(self, tool_input: Dict) -> str:
        raise NotImplementedError("WeatherTool only supports async operations")