import os
import asyncio
from langchain import LLMChain, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncGenerator
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class GeminiAgent:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        # Initialize callback handler
        self.callback = AsyncIteratorCallbackHandler()

        # Cache LLM instance
        self.llm = self._initialize_llm(api_key)

        # Initialize prompt templates and chains
        self._initialize_prompts()
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
            callbacks=[self.callback]
        )

    def _initialize_prompts(self):
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
                "Answer:"
            )
        )

        self.final_prompt = PromptTemplate(
            input_variables=["chain_of_thought"],
            template=(
                "Based on the chain-of-thought reasoning provided below, generate a final, concise, and factually accurate answer in proper markdown format.\n\n"
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

    async def run(self, query: str, reasoning: bool = False) -> AsyncGenerator[str, None]:
        try:
            if reasoning:
                # --- Stream chain-of-thought reasoning ---
                task = asyncio.create_task(self.cot_chain.arun(question=query))
                async for token in self.callback.aiter():
                    yield token
                # Signal completion of reasoning tokens
                self.callback.done.set()
                await task  # Wait for chain-of-thought to finish

                yield "\n--- Final Answer ---\n"

                # --- Stream final answer ---
                self.callback = AsyncIteratorCallbackHandler()  # Reset callback
                self.llm.callbacks = [self.callback]
                # Use the output of the previous task as the chain-of-thought (if needed)
                cot_result = task.result()
                task_final = asyncio.create_task(self.final_chain.arun(chain_of_thought=cot_result))
                async for token in self.callback.aiter():
                    yield token
                self.callback.done.set()
                await task_final

            else:
                # --- Direct response ---
                task = asyncio.create_task(self.direct_chain.arun(question=query))
                async for token in self.callback.aiter():
                    yield token
                self.callback.done.set()
                await task

        except Exception as e:
            logger.error(f"Gemini agent error: {str(e)}")
            yield f"Error: {str(e)}"
