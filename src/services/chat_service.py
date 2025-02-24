from typing import List, Dict, Union
import logging
from src.ai_model.gpt40 import GitHubGPTAgent
from src.core.prompts import SystemPrompts
from src.ai_model.gemini import GeminiAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        # Cache agents and prompts
        self._agents: Dict[str, Union[GitHubGPTAgent, GeminiAgent]] = {}
        self.default_agent = "gpt-4o"
        
    def get_agent(self, agent_type: str):
        """Get or create agent instance with caching."""
        if agent_type not in self._agents:
            if agent_type == "gemini":
                self._agents[agent_type] = GeminiAgent()
            else:
                self._agents[agent_type] = GitHubGPTAgent()
        return self._agents[agent_type]
        
    async def chat(self, messages: List[dict], agent_type: str = None) -> Dict[str, str]:
        try:
            if not messages:
                return {"error": "No messages provided."}
            
            agent_type = agent_type or self.default_agent
            agent = self.get_agent(agent_type)
            
            # Debug logging for system prompt
            system_prompt = SystemPrompts.get_prompt()
            logger.info(f"Using {agent_type} prompt: {system_prompt[:100]}...")  # Log first 100 chars
            
            if not messages[0].get("role") == "system":
                messages.insert(0, {"role": "system", "content": system_prompt})
                logger.info(f"Added system prompt to messages. Total messages: {len(messages)}")
            
            response_text = await agent.process_message(messages[-1].get("content", ""), messages)
            
            # Include model information in response
            model_info = {
                "gpt-4o": "GPT-40",
                "gemini": "Gemini 1.5 Flash",
            }
            
            return {
                "response": response_text,
                "model": model_info.get(agent_type, "Unknown Model"),
                "agent_type": agent_type
            }
            
        except Exception as e:
            logger.error(f"Chat service error: {str(e)}")
            return {
                "error": f"Chat service error: {str(e)}",
                "agent_type": agent_type
            }
