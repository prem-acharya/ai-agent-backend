import os
from openai import OpenAI

class DeepSeekAgent:
    def __init__(self):
        # Get DeepSeek API key from environment
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in your environment.")

        # Initialize the DeepSeek client
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )

    async def process_message(self, message: str) -> str:
        """Process a chat message using the DeepSeek API."""
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant powered by DeepSeek."},
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
            error_message = str(e)
            if "Insufficient Balance" in error_message:
                return "Error: Your DeepSeek API account has insufficient balance. Please add credits to your account."
            return f"Error processing message: {error_message}"
