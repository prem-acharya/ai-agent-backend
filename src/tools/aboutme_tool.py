from langchain.tools import BaseTool
from typing import ClassVar

class AboutMeTool(BaseTool):
    name: ClassVar[str] = "aboutme"
    description: ClassVar[str] = "Provides information about the user."

    def _run(self) -> str:
        """Returns a predefined response about the user."""
        return ("I am Prem Acharya, a full-stack developer who enjoys creating functional and user-friendly applications. "
                "I use modern web technologies like Next.js, React.js, and Tailwind CSS to develop solutions that solve real problems. "
                "Currently, I am involved in AI automation. You can view my portfolio at https://premacharya.vercel.app.")

    async def _arun(self) -> str:
        """Asynchronous version of the tool (if needed)."""
        return self._run()
