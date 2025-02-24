import os
from openai import OpenAI
from typing import List

class OpenAIAgent:
    def __init__(self):
        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in your environment.")

        # Initialize the OpenAI client
        self.client = OpenAI(
            api_key=api_key
        )

    async def process_message(self, message: str) -> str:
        """Process a chat message using the OpenAI API."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=2000,
                stream=False
            )
            if hasattr(response.choices[0].message, 'content'):
                return response.choices[0].message.content
            return "No response generated"
        except Exception as e:
            return f"Error processing message: {str(e)}" 