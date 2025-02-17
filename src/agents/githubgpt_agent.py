import os
from openai import OpenAI, AsyncOpenAI
from typing import List

class GitHubGPTAgent:
    def __init__(self):
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN not found in your environment.")

        self.client = AsyncOpenAI(
            api_key=github_token,
            base_url="https://models.inference.ai.azure.com"
        )

    async def process_message(self, message: str) -> str:
        """Process a chat message using the GitHub GPT API."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant powered by GitHub GPT."},
                    {"role": "user", "content": message}
                ],
                temperature=1,
                max_tokens=4096,
                top_p=1,
                stream=False
            )
            if hasattr(response.choices[0].message, 'content'):
                return response.choices[0].message.content
            return "No response generated"
        except Exception as e:
            return f"Error processing message: {str(e)}" 