from langchain_core.prompts import PromptTemplate

def initialize_prompts():
    cot_prompt = PromptTemplate(
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

    direct_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "Provide a direct, concise answer in proper markdown format with relevant emojis for: {question}\n\n"
            "Use **bold**, *italics*, `inline code`, and other markdown elements correctly.\n\n"
            "Answer:"
        )
    )

    final_prompt = PromptTemplate(
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

    return cot_prompt, direct_prompt, final_prompt 