from typing import List, Dict
from src.agents.githubgpt_agent import GitHubGPTAgent
from src.core.prompts import SystemPrompts

class ChatService:
    def __init__(self):
        # Cache agents and prompts
        self._agents: Dict[str, GitHubGPTAgent] = {}
        self.default_agent = "githubgpt"
        
    def get_agent(self, agent_type: str) -> GitHubGPTAgent:
        """Get or create agent instance with caching."""
        if agent_type not in self._agents:
            self._agents[agent_type] = GitHubGPTAgent()
        return self._agents[agent_type]
        
    async def chat(self, messages: List[dict], agent_type: str = None) -> str:
        try:
            if not messages:
                return "No messages provided."
            
            latest_message = messages[-1].get("content", "")
            if not latest_message:
                return "Message content is empty."
            
            agent_type = agent_type or self.default_agent
            agent = self.get_agent(agent_type)
            
            # Only add system prompt if needed
            if not messages[0].get("role") == "system":
                system_prompt = SystemPrompts.get_prompt(agent_type)
                messages.insert(0, {"role": "system", "content": system_prompt})
                
            return await agent.process_message(latest_message, messages)
        except Exception as e:
            return f"Chat service error: {str(e)}"
