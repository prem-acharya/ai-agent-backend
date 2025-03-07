import os
import asyncio
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from src.tools.weather.weather_tool import WeatherTool
from src.tools.aboutme_tool import AboutMeTool
from src.tools.datetime.time_tool import CurrentTimeTool
from src.tools.websearch.websearch_tool import WebSearchTool
from typing import AsyncGenerator
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import AsyncIteratorCallbackHandler

class ToolTrackingCallback(BaseCallbackHandler):
    def __init__(self, tools_used_set):
        self.tools_used = tools_used_set

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        if "name" in serialized:
            self.tools_used.add(serialized["name"])

class GitHubGPTAgent:
    def __init__(self):
        self.callback = AsyncIteratorCallbackHandler()
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4o",
            model_name="gpt-4o",
            api_version="2024-02-15-preview",
            temperature=0.7,
            max_tokens=4096,
            streaming=True,
            callbacks=[self.callback]
        )

        # tools
        self.base_tools = [WeatherTool(), AboutMeTool(), CurrentTimeTool()]

        self.cot_prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "You are a highly advanced reasoning assistant that harnesses the latest capabilities "
                "from DeepSeek, OpenAI, GPT‑latest, and Glork 2. Please provide your internal chain‑of‑thought "
                "reasoning for the following question in clear, coherent paragraphs, using the same language as the user's question. "
                "Do not include the final answer here—only your internal reasoning.\n\n"
                "Question: {question}\n\n"
                "Chain-of-Thought Reasoning (in paragraphs):"
            )
        )

        self.direct_prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "Provide a direct, concise answer in proper markdown format with relevant emojis for: {question}\n\n"
                "Use **bold**, *italics*, `inline code`, and other markdown elements correctly.\n\n"
                "Answer:"
            )
        )

        self.final_prompt = PromptTemplate(
            input_variables=["chain_of_thought"],
            template=(
                "Based on the chain-of-thought reasoning provided below, generate a final, concise, and factually accurate answer in the same language (proper language use) as the user's question. "
                "Present the final answer in proper markdown format with minimal, relevant emojis to enhance clarity.\n\n"
                "Use **bold**, *italics*, `inline code`, and other markdown elements correctly.\n\n"
                "Chain-of-Thought Reasoning:\n"
                "{chain_of_thought}\n\n"
                "Final Answer:"
            )
        )

        self.tools_used = set()

    async def run(self, query: str, reasoning: bool = False, websearch: bool = False, city: str = "Kolkata") -> AsyncGenerator[str, None]:
        try:
            self.tools_used = set()
            tools = self.base_tools.copy()
            
            # Track current time tool
            current_time_tool = CurrentTimeTool()
            current_time = await current_time_tool._arun(city)
            self.tools_used.add("current time")

            # Check for weather-related query
            if any(word in query.lower() for word in ["weather", "temperature", "humidity", "wind", "vatavaran", "hava"]):
                self.tools_used.add("weather")
                
            if any(word in query.lower() for word in ["about me", "bio", "background", "education", "experience", "skills", "about owner", "about creator", "about developer", "about author"]):
                self.tools_used.add("about me")

            if websearch:
                web_tool = WebSearchTool()
                tools.append(web_tool)
                self.tools_used.add("web search")
                query = f"{query} (as of {current_time})"

            # Initialize agent with proper callback handler
            callback_handler = ToolTrackingCallback(self.tools_used)
            agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=False,
                handle_parsing_errors=True,
                callbacks=[callback_handler]
            )

            if websearch and reasoning:
                # --- Web search phase ---
                yield "Gathering web search results...\n"
                web_search_result = ""
                async for chunk in agent.astream(f"Search and gather current information about: {query}"):
                    text = chunk.get("output", "") if isinstance(chunk, dict) else str(chunk)
                    web_search_result += text
                    yield text
                yield "\n"

                # --- Stream reasoning ---
                yield "\nChain-of-Thought Reasoning:\n"
                reasoning_text = ""
                prompt = self.cot_prompt.format(question=f"{query}\n\nBased on this web search information:\n{web_search_result}")
                async for chunk in agent.astream(prompt):
                    text = chunk.get("output", "") if isinstance(chunk, dict) else str(chunk)
                    reasoning_text += text
                    yield text

                yield "\n--- Final Answer ---\n"
                
                # --- Stream final answer ---
                final_prompt = self.final_prompt.format(
                    chain_of_thought=f"Web Search Results:\n{web_search_result}\n\nReasoning:\n{reasoning_text}"
                )
                async for chunk in agent.astream(final_prompt):
                    text = chunk.get("output", "") if isinstance(chunk, dict) else str(chunk)
                    yield text

            elif reasoning:
                # --- Reasoning only ---
                yield "Chain-of-Thought Reasoning:\n"
                reasoning_text = ""
                prompt = self.cot_prompt.format(question=query)
                async for chunk in agent.astream(prompt):
                    text = chunk.get("output", "") if isinstance(chunk, dict) else str(chunk)
                    reasoning_text += text
                    yield text

                yield "\n--- Final Answer ---\n"
                final_prompt = self.final_prompt.format(chain_of_thought=reasoning_text)
                async for chunk in agent.astream(final_prompt):
                    text = chunk.get("output", "") if isinstance(chunk, dict) else str(chunk)
                    yield text

            else:
                # --- Direct response ---
                async for chunk in agent.astream(self.direct_prompt.format(question=query)):
                    text = chunk.get("output", "") if isinstance(chunk, dict) else str(chunk)
                    if text.strip():
                        yield text

        except Exception as e:
            yield f"Error: {str(e)}"

    async def _stream_reasoning(self, agent, query, web_search_result=""):
        # This async generator yields tokens for reasoning.
        reasoning_text = ""
        prompt = self.cot_prompt.format(question=f"{query}\n\nBased on this web search information:\n{web_search_result}")
        async for chunk in agent.astream(prompt):
            text = chunk.get("output", "") if isinstance(chunk, dict) else str(chunk)
            reasoning_text += text
            yield text
        # Removed 'return reasoning_text' because async generators cannot return a value.

    async def _stream_final_answer(self, agent, web_search_result="", reasoning_text=""):
        # This async generator yields tokens for the final answer.
        final_prompt = self.final_prompt.format(
            chain_of_thought=f"Web Search Results:\n{web_search_result}\n\nReasoning:\n{reasoning_text}"
        )
        async for chunk in agent.astream(final_prompt):
            text = chunk.get("output", "") if isinstance(chunk, dict) else str(chunk)
            yield text
