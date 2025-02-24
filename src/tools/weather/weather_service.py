import os
import httpx
from typing import Dict

class WeatherService:
    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")
        if not self.api_key:
            raise ValueError("WEATHER_API_KEY not found in environment variables")
        
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    async def get_weather_by_city(
        self, 
        city: str,
        units: str = "metric",
        lang: str = "en"
    ) -> Dict:
        try:
            params = {
                "q": city,
                "appid": self.api_key,
                "units": units,
                "lang": lang
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.base_url,
                    params=params,
                    timeout=10.0
                )
                
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPError as e:
            error_msg = f"HTTP error occurred: {str(e)}"
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error fetching weather data: {str(e)}"
            raise ValueError(error_msg) 