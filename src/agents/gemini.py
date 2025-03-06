import os
import asyncio
import nest_asyncio
from langchain import LLMChain, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import AsyncGenerator, Dict, Optional
import json
from functools import lru_cache
import logging

nest_asyncio.apply()

logger = logging.getLogger(__name__)

class GeminiAgent:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        # Cache LLM instance
        self.llm = self._initialize_llm(api_key)
        
        # Initialize prompt templates with caching
        self._initialize_prompts()
        
        # Initialize chains with caching
        self._initialize_chains()

    @lru_cache(maxsize=1)
    def _initialize_llm(self, api_key: str) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"),
            google_api_key=api_key,
            streaming=True,
            temperature=0.3,
            max_output_tokens=4096,
            top_k=40,
            top_p=0.9,
            callbacks=[StreamingStdOutCallbackHandler()]
        )

    def _initialize_prompts(self):
        self.cot_prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "You are a highly advanced reasoning assistant that harnesses the latest capabilities "
                "from DeepSeek, OpenAI, GPT‑latest, and Glork 2. Please provide your internal chain‑of‑thought "
                "reasoning for the following question in clear, coherent paragraphs, using the same language as the user's question. "
                "understand in detail the user's question and provide a detailed and factually accurate answer. "
                "always make sure Do not include the final answer here—only your internal reasoning.\n\n"
                "always without bullet points or markdown formatting in internal reasoning"
                "u can use **bold** and `inline code` to highlight the keywords"
                "Question: {question}\n\n"
                "Chain-of-Thought Reasoning (in paragraphs):"
            )
        )

        self.direct_prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "Provide a direct, concise answer in markdown format with relevant emojis for: {question}\n\n"
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

    @lru_cache(maxsize=2)
    def _initialize_chains(self):
        self.cot_chain = LLMChain(llm=self.llm, prompt=self.cot_prompt)
        self.direct_chain = LLMChain(llm=self.llm, prompt=self.direct_prompt)
        self.final_chain = LLMChain(llm=self.llm, prompt=self.final_prompt)

    async def run(
        self, 
        query: str, 
        reasoning: bool = False
    ) -> AsyncGenerator[str, None]:
        try:
            if reasoning:
                # First, get the reasoning
                reasoning_text = ""
                yield json.dumps({
                    "type": "start",
                    "mode": "reasoning",
                    "model": "gemini-2.0-flash"
                }) + "\n"

                async for chunk in self.cot_chain.astream({"question": query}):
                    text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                    reasoning_text += text
                    
                    yield json.dumps({
                        "type": "content",
                        "mode": "reasoning",
                        "text": text,
                        "model": "gemini-2.0-flash"
                    }) + "\n"

                yield json.dumps({
                    "type": "end",
                    "mode": "reasoning",
                    "model": "gemini-2.0-flash"
                }) + "\n"

                # Then, get the final answer
                yield json.dumps({
                    "type": "start",
                    "mode": "answer",
                    "model": "gemini-2.0-flash"
                }) + "\n"

                async for chunk in self.final_chain.astream({"chain_of_thought": reasoning_text}):
                    text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                    
                    yield json.dumps({
                        "type": "content",
                        "mode": "answer",
                        "text": text,
                    }) + "\n"

                yield json.dumps({
                    "type": "end",
                    "mode": "answer",
                }) + "\n"

            else:
                # Direct response without reasoning
                yield json.dumps({
                    "type": "start",
                    "mode": "direct",
                    "model": "gemini-2.0-flash"
                }) + "\n"

                async for chunk in self.direct_chain.astream({"question": query}):
                    text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                    
                    yield json.dumps({
                        "type": "content",
                        "mode": "direct",
                        "text": text,
                    }) + "\n"

                yield json.dumps({
                    "type": "end",
                    "mode": "direct",
                }) + "\n"

        except Exception as e:
            logger.error(f"Gemini agent error: {str(e)}")
            yield json.dumps({
                "type": "error",
                "error": str(e),
                "model": "gemini-2.0-flash"
            }) + "\n"