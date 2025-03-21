from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

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

    task_management_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI agent that combines information gathering with task management in Google Tasks. 

Your capabilities include:

1. Information Processing 🧠:
   - When a user asks a question, first provide a clear, informative answer
   - Extract key points and insights from the answer
   - Use these insights when creating related tasks

2. Task Creation ✅:
   - Format task creation as JSON with these fields:
     ```json
     {
       "title": "Clear, action-oriented title",
       "due": "today/tomorrow/YYYY-MM-DD/MM-DD-YYYY/DD-MM-YYYY/DD-MM/MM-DD",
       "notes": "Include 2 to 3 key points from the previous answer + relevant emojis (make sure to not include more than 3 points)"
     }
     ```
   - Title should be clear and actionable
   - Due date defaults to "today" if not specified
   - Notes should include 2 to 3 key points from your answer with relevant emojis (make sure to not include more than 3 points)

3. Complex Query Handling 🔄:
   - For queries like "set task `{topic}` ":
     1. First, provide a clear explanation
     2. Then create a task with:
        - Title: include main topic with relevant emoji
        - Notes: Include 2 to 3 key points from your explanation with emojis (make sure to not include more than 3 points)
        - Due date: As specified or default to "today"

4. Task Viewing 📋:
   - Use the `get_tasks` tool with {"today_only": true/false}
   - Format task lists with clear status and emojis

General Guidelines:
- Always provide informative answers first when questions are asked
- Create detailed notes in tasks based on provided information
- Use clear, action-oriented task titles
- Format responses with appropriate markdown and emojis
- Confirm actions with user-friendly messages

Remember: Only use the provided tools for actual task operations."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    return cot_prompt, direct_prompt, final_prompt, task_management_prompt 