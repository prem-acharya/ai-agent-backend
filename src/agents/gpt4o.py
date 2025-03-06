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
        """Runs the AI agent with optional web search support and includes response metadata."""
        tools = self.base_tools.copy()
        formatted_prompt = self.prompt_template.format(query=query)
        current_time, current_date = "", ""

        # First, get the current date and time
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
            tools.append(WebSearchTool())

        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True
        )

        async def stream_response():
            """Generator to stream AI response in real time with metadata."""
            start_time = time.time()
            collected_response = ""
            try:
                async for chunk in agent.astream(formatted_prompt):
                    # Ensure the chunk is a string
                    if hasattr(chunk, 'content'):
                        chunk_str = chunk.content
                    elif isinstance(chunk, dict):
                        chunk_str = str(chunk.get('content', str(chunk)))
                    else:
                        chunk_str = str(chunk)
                    collected_response += chunk_str
                    yield chunk_str

                # Calculate processing time
                processing_time = time.time() - start_time

                # Get token usage if available; fallback to "N/A"
                token_usage = getattr(self.llm, "last_token_usage", "N/A")

                # Prepare metadata
                metadata = {
                    "Model Information": self.llm.model_name,
                    "Tool Usage": [tool.__class__.__name__ for tool in tools],
                    "Token Usage": token_usage,
                    "Processing Time": processing_time
                }

                # Yield the metadata as a final JSON chunk
                metadata_chunk = "\n\n---METADATA---\n" + json.dumps(metadata) + "\n"
                yield metadata_chunk

            except Exception as e:
                yield f"⚠️ **Error running agent:** `{str(e)}`"

        return stream_response()  # Returns a generator for streaming
