import os
import json
import httpx
from typing import List, Dict

class GeminiAgent:
    def __init__(self):
        # Get Gemini API key from environment
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in your environment.")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.model = "gemini-1.5-flash"
        
    async def process_message(self, message: str, messages: List[dict] = None) -> str:
        """Process a chat message using the Gemini API."""
        try:
            # Prepare the API endpoint
            url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
            
            # Convert messages to Gemini format
            content = {
                "contents": [{
                    "parts": [{"text": message}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 4096,  # Maximum token limit
                    "topP": 0.9,
                    "topK": 40,
                    "candidateCount": 1,
                }
            }
            
            # If there are system messages, include them
            if messages and messages[0].get("role") == "system":
                content["contents"][0]["parts"].insert(0, {
                    "text": messages[0].get("content", "")
                })
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=content,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    error_msg = response.json().get("error", {}).get("message", "Unknown error")
                    return f"Error from Gemini API: {error_msg}"
                
                response_data = response.json()
                if "candidates" in response_data:
                    return response_data["candidates"][0]["content"]["parts"][0]["text"]
                return "No response generated"
                
        except Exception as e:
            error_message = str(e)
            if "quota exceeded" in error_message.lower():
                return "Error: Gemini API quota exceeded. Please check your API limits."
            return f"Error processing message: {error_message}"
