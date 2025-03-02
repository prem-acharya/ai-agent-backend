from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from src.tools.weather.weather_tool import WeatherTool
from src.tools.aboutme_tool import AboutMeTool
from src.tools.datetime.time_tool import CurrentTimeTool
from src.tools.websearch.websearch_tool import WebSearchTool
import os

class GitHubGPTAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4o",
            model_name="gpt-4o",
            api_version="2024-02-15-preview",
            temperature=0.7
        )

        # Default tools (without web search)
        self.base_tools = [WeatherTool(), AboutMeTool(), CurrentTimeTool()]

        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template=(
                "You are an AI assistant. Always respond in **Markdown format** with proper headers, lists, and code blocks when necessary.\n"
                "Use **bold**, *italics*, `inline code`, and other markdown elements correctly.\n"
                "Always Include emojis 😊 to make responses engaging.\n\n"
                "### User Query:\n"
                "{query}\n\n"
                "### AI Response (Markdown format with emojis):"
            ),
        )

    async def run(self, query: str, websearch: bool = False, city: str = "Kolkata") -> dict:
        """Runs the AI agent with optional web search support."""
        tools = self.base_tools.copy()

        # Format the query using the prompt template
        formatted_prompt = self.prompt_template.format(query=query)

        current_time = ""
        current_date = ""

        # First, get the current date and time
        time_tool = CurrentTimeTool()
        current_time_response = time_tool._run(city)  # Fetch current time

        try:
            time_parts = current_time_response.split(": ", 1)[-1].strip()

            # Extract date and time separately
            if "T" in time_parts:
                date_part, time_part = time_parts.split("T", 1)  # Ensures only one split
                current_date = date_part.strip()
                
                if "+" in time_part:  # Ensure it handles time zones correctly
                    current_time = time_part.split("+", 1)[0].strip()
                else:
                    current_time = time_part.strip()
        except Exception as e:
            print(f"Error parsing time: {e}")

        # If web search is enabled, modify the query
        if websearch:
            formatted_prompt = f"{formatted_prompt} (as of {current_date}, {current_time})"
            tools.append(WebSearchTool())

        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS, # Multi-tool support
            # agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, Multi-tool support also works with gemini
            verbose=True
        )

        try:
            result = await agent.arun(formatted_prompt)

            # Detect tools used in response
            used_tools = []
            if "temperature" in result.lower() or "weather" in result.lower():
                used_tools.append("weather")
            if "software developer" in result.lower():
                used_tools.append("about me")
            if "current time" in result.lower() or "time" in result.lower() or "date" in result.lower() or "latest" in result.lower():
                used_tools.append("current time")
            if websearch:
                used_tools.append("web search")

            return {
                "response": result,
                "error": None,
                "agent": ", ".join(used_tools),  # Show multiple tools used
                "model": "gpt-4o"
            }
        except Exception as e:
            return {
                "response": f"Error running agent: {str(e)}",
                "error": str(e),
                "agent": "",
                "model": "gpt-4o"
            }
