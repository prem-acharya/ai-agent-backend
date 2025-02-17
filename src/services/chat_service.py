from typing import List
from src.agents.githubgpt_agent import GitHubGPTAgent

class ChatService:
    def __init__(self):
        self.agents = {
            "githubgpt": GitHubGPTAgent()
        }
        self.default_agent = "githubgpt"
        
    async def chat(self, messages: List[dict], agent_type: str = None) -> str:
        try:
            if not messages:
                return "No messages provided."
            
            latest_message = messages[-1].get("content")
            if not latest_message:
                return "Message content is empty."
            
            # Use specified agent or default to GitHubGPT
            agent = self.agents.get(agent_type or self.default_agent)
            if not agent:
                return f"Invalid agent type: {agent_type}"
                
            return await agent.process_message(latest_message)
        except Exception as e:
            return f"Chat service error: {str(e)}"
