from typing import List, Dict, Union
import logging
import re
from models.gpt40 import GitHubGPTAgent
from models.gemini import GeminiAgent
from src.tools.weather.weather_tool import WeatherTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self._agents: Dict[str, Union[GitHubGPTAgent, GeminiAgent]] = {}
        self.default_agent = "gpt-4o"
        self.weather_tool = WeatherTool()

    def get_agent(self, agent_type: str):
        """Get or create agent instance with caching."""
        if agent_type not in self._agents:
            self._agents[agent_type] = self.create_agent(agent_type)
        return self._agents[agent_type]

    def create_agent(self, agent_type: str):
        """Create a new agent based on the agent type."""
        if agent_type == "gemini":
            return GeminiAgent()
        return GitHubGPTAgent()

    async def detect_weather_query(self, message: str) -> Dict[str, str]:
        """Detect if the message is a weather query."""
        weather_patterns = self.get_weather_patterns()
        for pattern in weather_patterns:
            match = re.search(pattern, message.lower())
            if match:
                city = match.group(1).strip()
                return {"city": city, "units": "metric"}
        return None

    def get_weather_patterns(self) -> List[str]:
        """Return a list of regex patterns for detecting weather queries."""
        return [
            r"weather\s+(?:in|at|for)?\s+([a-zA-Z\s]+)",
            r"temperature\s+(?:in|at|for)?\s+([a-zA-Z\s]+)",
            r"how(?:'s| is) the weather (?:in|at|for)?\s+([a-zA-Z\s]+)",
            r"what(?:'s| is) the weather (?:like )?(?:in|at|for)?\s+([a-zA-Z\s]+)",
            r"humidity\s+(?:in|at|for)?\s+([a-zA-Z\s]+)",
            r"conditions\s+(?:in|at|for)?\s+([a-zA-Z\s]+)"
        ]

    async def chat(self, messages: List[dict], agent_type: str = None) -> Dict[str, str]:
        """Handle chat messages and return responses."""
        try:
            if not messages:
                return {"error": "No messages provided."}

            current_message = messages[-1].get("content", "")
            weather_query = await self.detect_weather_query(current_message)

            if weather_query:
                return await self.handle_weather_query(weather_query)

            return await self.handle_regular_chat(messages, agent_type)

        except Exception as e:
            logger.error(f"Chat service error: {str(e)}")
            return {
                "error": f"Chat service error: {str(e)}",
                "agent_type": agent_type
            }

    async def handle_weather_query(self, weather_query: Dict[str, str]) -> Dict[str, str]:
        """Handle weather queries and return weather information."""
        weather_info = await self.weather_tool._arun(weather_query)
        model_info = self.get_model_info().get("weather", "gpt-4o")
        
        return {
            "response": weather_info,
            "model": model_info,
            "agent_type": "weather"
        }

    async def handle_regular_chat(self, messages: List[dict], agent_type: str) -> Dict[str, str]:
        """Handle regular chat messages and return responses from the agent."""
        agent_type = agent_type or self.default_agent
        agent = self.get_agent(agent_type)

        response_text = await agent.process_message(messages[-1].get("content", ""), messages)
        model_info = self.get_model_info()

        return {
            "response": response_text,
            "model": model_info.get(agent_type, "Unknown Model"),
            "agent_type": agent_type
        }

    def get_model_info(self) -> Dict[str, str]:
        """Return a dictionary of model information."""
        return {
            "gpt-4o": "GPT-40",
            "gemini": "Gemini 1.5 Flash",
        }

    async def get_weather(self, city: str, units: str = "metric") -> str:
        """Get weather information for a specific city."""
        try:
            weather_info = await self.weather_tool._arun({
                "city": city,
                "units": units
            })
            return weather_info
        except Exception as e:
            logger.error(f"Weather service error: {str(e)}")
            return f"Error getting weather: {str(e)}"
