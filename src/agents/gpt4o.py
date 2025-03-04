from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from src.tools.weather.weather_tool import WeatherTool
from src.tools.aboutme_tool import AboutMeTool
from src.tools.datetime.time_tool import CurrentTimeTool
from src.tools.websearch.websearch_tool import WebSearchTool
import os
import time

class GitHubGPTAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_key="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",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4o",
            model_name="gpt-4o",
            api_version="2024-02-15-preview",
            temperature=0.7,
            streaming=True
        )

        # Default tools (without web search)
        self.base_tools = [WeatherTool(), AboutMeTool(), CurrentTimeTool()]

        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template=(
                "You are an agent pro. Always respond in **Markdown format** with proper headers, lists, and code blocks when necessary.\n\n"
                "Use **bold**, *italics*, `inline code`, and other markdown elements correctly.\n\n"
                "Always use relevant emojis 😊 to make responses engaging.\n\n"
                "### User Query:\n\n"
                "{query}\n\n"
                "### AI Response (Markdown format with emojis):"
            ),
        )

    async def run(self, query: str, websearch: bool = False, city: str = "Kolkata") -> dict:
        """Runs the AI agent with optional web search support."""
        tools = self.base_tools.copy()
        formatted_prompt = self.prompt_template.format(query=query)
        current_time ,current_date = "", ""

        # First, get the current date and time
        time_tool = CurrentTimeTool()
        current_time_response = time_tool._run(city)  # Fetch current time

        try:
            time_response = CurrentTimeTool()._run(city)
            time_parts = time_response.split(": ", 1)[-1].strip()
            if "T" in time_parts:
                current_date, time_part = time_parts.split("T", 1)
                current_time = time_part.split("+")[0].strip() if "+" in time_part else time_part.strip()
        except Exception as e:
            print(f"⚠️ Error parsing time: {e}")

        if websearch:
            formatted_prompt = f"{formatted_prompt} (as of {current_date}, {current_time})"
            tools.append(WebSearchTool())

        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            
            verbose=True
        )

        async def stream_response():
            """Generator to stream AI response in real time."""
            try:
                async for chunk in agent.astream(formatted_prompt):
                    # Convert the chunk to string if it's a dictionary
                    if hasattr(chunk, 'content'):
                        yield chunk.content
                    elif isinstance(chunk, dict):
                        yield str(chunk.get('content', str(chunk)))
                    else:
                        yield str(chunk)
            except Exception as e:
                yield f"⚠️ **Error running agent:** `{str(e)}`"

        return stream_response() # Returns a generator for streaming
