from langchain.agents import initialize_agent, AgentType
from langchain_openai import AzureChatOpenAI
from src.tools.weather.weather_tool import WeatherTool
from src.tools.aboutme_tool import AboutMeTool
from src.tools.time_tool import CurrentTimeTool
import os

print(os.getenv("AZURE_OPENAI_API_KEY"))

class GitHubGPTAgent:

    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_key= "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",
            azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name= "gpt-4o",
            model_name="gpt-4o",
            api_version="2024-02-15-preview",
            temperature=0.7)

        self.tools = [WeatherTool(), AboutMeTool(), CurrentTimeTool()]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS, # ✅ Multi-tool support also works with gemini
            # agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, ✅ Multi-tool support also works with gemini
            verbose=True)

    async def run(self, query: str) -> dict:
        try:
            result = await self.agent.arun(query)

            # Initialize used_tools list
            used_tools = []
                
            if "temperature" in result.lower() or "weather" in result.lower():
                used_tools.append("weather")
            if "software developer" in result.lower():
                used_tools.append("about_me")
            if "time" in result.lower():
                used_tools.append("current_time")

            return {
                "response": result,
                "error": None,
                "agent": ", ".join(used_tools),
                "model": self.llm.deployment_name
            }
        except Exception as e:
            return {
                "response": f"Error running agent: {str(e)}",
                "error": str(e),
                "agent": "",
                "model": self.llm.deployment_name
            }
