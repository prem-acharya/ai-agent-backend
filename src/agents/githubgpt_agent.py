import os
from openai import AsyncOpenAI
from typing import List

class GitHubGPTAgent:
    def __init__(self):
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN not found in your environment.")

        # Initialize client once during instantiation
        self.client = AsyncOpenAI(
            api_key=github_token,
            base_url="https://models.inference.ai.azure.com",
            timeout=30.0  # Set timeout to 30 seconds
        )

    async def process_message(self, message: str, messages: List[dict]) -> str:
        """Process a chat message using the GitHub GPT API."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,  # Lower temperature for faster, more focused responses
                max_tokens=2048,  # Reduced from 4096 for faster responses
                top_p=0.9,
                presence_penalty=0,
                frequency_penalty=0,
                stream=False
            )
            return response.choices[0].message.content or "No response generated"
        except Exception as e:
            return f"Error processing message: {str(e)}" 