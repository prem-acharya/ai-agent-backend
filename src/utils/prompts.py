"""Prompt templates for various AI interactions."""

from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.utils.prompt.task_prompts import get_task_management_prompt

def initialize_prompts():
    """Initialize all prompt templates."""
    cot_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are a highly advanced reasoning assistant that harnesses the latest capabilities "
            "from DeepSeek, OpenAI, GPT‑latest, and Glork 2. Please provide your internal chain‑of‑thought "
            "reasoning for the following question in clear, coherent paragraphs, using the same language as the user's question. "
            "understand in detail the user's question and provide a detailed and factually accurate answer. "
            "always make sure Do not include the final answer here—only your internal reasoning.\n\n"
            "always without bullet points or markdown formatting in internal reasoning"
            "u can only use **bold** and `inline code` to highlight the keywords (no other markdown formatting is allowed)"
            "Question: {question}\n\n"
            "Chain-of-Thought Reasoning (in paragraphs):"
        )
    )

    direct_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "Provide a direct, concise answer in proper markdown format with relevant emojis for: {question}\n\n"
            "Use proper markdown formatting **bold**, *italics*, `inline code`, and other markdown elements correctly.\n\n"
            "Answer:"
        )
    )

    final_prompt = PromptTemplate(
        input_variables=["chain_of_thought", "web_context"],
        template=(
            "Based on the following chain-of-thought reasoning and web search context (if provided), "
            "generate a final, concise, and factually accurate answer in proper markdown format "
            "Use proper markdown formatting **bold**, *italics*, `inline code`, and other markdown elements correctly.\n\n"
            "with relevant emojis.\n\n"
            "Chain-of-Thought Analysis:\n{chain_of_thought}\n\n"
            "{web_context}\n\n"
            "Final Answer:"
        )
    )

    task_management_prompt = get_task_management_prompt()

    return cot_prompt, direct_prompt, final_prompt, task_management_prompt 