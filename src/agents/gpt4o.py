from langchain.agents import agent_types, initialize_agent, AgentType
from langchain_openai import AzureChatOpenAI
from src.tools.weather.weather_tool import WeatherTool
import os

class GitHubGPTAgent:

    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_key= os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name= "gpt-4o",
            model_name="gpt-4o",
            api_version="2024-02-15-preview",
            temperature=0.7)

        self.tools = [WeatherTool()]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True)

    async def run(self, query: str) -> dict:
        try:
            result = await self.agent.arun(query)

            # Check if weather information is present in the response
            weather_indicators = [
                "temperature", "humidity", "wind", "visibility", "weather",
                "forecast"
            ]
            is_weather_used = any(indicator in result.lower()
                                  for indicator in weather_indicators)

            return {
                "response": result,
                "error": None,
                "agent": "weather" if is_weather_used else "",
                "model": self.llm.deployment_name
            }
        except Exception as e:
            return {
                "response": f"Error running agent: {str(e)}",
                "error": str(e),
                "agent": "",
                "model": self.llm.deployment_name
            }
