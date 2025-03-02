import os
import requests
from langchain.tools import BaseTool
from typing import ClassVar

class WebSearchTool(BaseTool):
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = "Searches the web for the latest information using Tavily AI."

    def _run(self, query: str) -> str:
        """Perform a web search using Tavily AI API results."""
        api_key = os.getenv("TAVILY_API_KEY")
        api_url = os.getenv("TAVILY_API_URL")

        if not api_key or not api_url:
            return "Error: Tavily API key or API URL is missing. Please set the environment variables."

        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            body = {"query": query}

            response = requests.post(api_url, json=body, headers=headers)
            response.raise_for_status()  # Raise an error for HTTP failures

            data = response.json()
            if "results" in data and data["results"]:
                results = data["results"][:5]
                formatted_results = []

                for result in results:
                    title = result.get("title", "No title available")
                    url = result.get("url", "No URL available")
                    content = result.get("content", "No content available")
                    source = result.get("source", "Unknown source")
                    image_url = result.get("image", None)  # ✅ Fetch image if available
                    
                    # ✅ Format results with an image if available
                    result_text = f"### **{title}**\n📌 Source: {source}\n🔗 [Read more]({url})\n\n{content}\n"
                    if image_url:
                        result_text += f"\n🖼 ![Image]({image_url})"

                    formatted_results.append(result_text)

                return "\n\n---\n\n".join(formatted_results)
            return "No relevant search results found."

        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version of the web search tool."""
        return self._run(query)
