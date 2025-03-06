from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from src.tools.weather.weather_tool import WeatherTool
from src.tools.aboutme_tool import AboutMeTool
from src.tools.datetime.time_tool import CurrentTimeTool
from src.tools.websearch.websearch_tool import WebSearchTool
import os
import time
import json
from typing import AsyncGenerator

class GitHubGPTAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_key="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",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4o",
            model_name="gpt-4o",
            api_version="2024-02-15-preview",
            temperature=0.7,
            max_tokens=4096,
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

    async def run(self, query: str, websearch: bool = False, city: str = "Kolkata") -> AsyncGenerator[str, None]:
        """Runs the AI agent with optional web search support."""
        try:
            tools = self.base_tools.copy()
            formatted_prompt = self.prompt_template.format(query=query)

            if websearch:
                tools.append(WebSearchTool())
                used_tools = ["WebSearchTool"]  # Track that the web search tool is used
            else:
                used_tools = []  # Initialize used_tools

            # Get the current date and time
            time_tool = CurrentTimeTool()
            try:
                time_response = time_tool._run(city)
                time_parts = time_response.split(": ", 1)[-1].strip()
                if "T" in time_parts:
                    current_date, time_part = time_parts.split("T", 1)
                    current_time = time_part.split("+")[0].strip() if "+" in time_part else time_part.strip()
            except Exception as e:
                print(f"⚠️ Error parsing time: {e}")

            if websearch:
                formatted_prompt = f"{formatted_prompt} (as of {current_date}, {current_time})"

            agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=False  # Set to False to hide intermediate steps
            )

            # Signal stream start
            yield json.dumps({
                "type": "start",
                "mode": "direct",
                "model": "gpt-4o"
            }) + "\n"

            # Stream the response
            final_response = ""
            async for chunk in agent.astream(formatted_prompt):
                if hasattr(chunk, 'output'):
                    final_response = chunk.output
                elif isinstance(chunk, dict) and 'output' in chunk:
                    final_response = chunk['output']
                elif isinstance(chunk, str):
                    final_response = chunk

                # Check for tools used in the response
                if "weather" in final_response.lower() or "temperature" in final_response.lower():
                    used_tools.append("WeatherTool")
                if "about me" in final_response.lower() or "about you" in final_response.lower() or "about my" in final_response.lower():
                    used_tools.append("AboutMeTool")
                if "current time" in final_response.lower() or "current date" in final_response.lower():
                    used_tools.append("CurrentTimeTool")

                if final_response:
                    # Only yield the final, cleaned response
                    response_data = {
                        "type": "content",
                        "text": final_response,
                        "model": "gpt-4o"
                    }
                    if used_tools:  # Only include used_tools if not empty
                        response_data["used_tools"] = list(set(used_tools))

                    yield json.dumps(response_data) + "\n"

            # Signal stream end
            yield json.dumps({
                "type": "end",
                "mode": "direct"
            }) + "\n"

        except Exception as e:
            yield json.dumps({
                "type": "error",
                "error": str(e),
                "model": "gpt-4o"
            }) + "\n"
