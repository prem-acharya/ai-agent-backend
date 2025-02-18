from typing import List, Dict
import logging
from src.agents.githubgpt_agent import GitHubGPTAgent
from src.core.prompts import SystemPrompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            
            agent_type = agent_type or self.default_agent
            agent = self.get_agent(agent_type)
            
            # Debug logging for system prompt
            system_prompt = SystemPrompts.get_prompt()
            logger.info(f"Using {agent_type} prompt: {system_prompt[:100]}...")  # Log first 100 chars
            
            if not messages[0].get("role") == "system":
                messages.insert(0, {"role": "system", "content": system_prompt})
                logger.info(f"Added system prompt to messages. Total messages: {len(messages)}")
            
            return await agent.process_message(messages[-1].get("content", ""), messages)
        except Exception as e:
            logger.error(f"Chat service error: {str(e)}")
            return f"Chat service error: {str(e)}"
