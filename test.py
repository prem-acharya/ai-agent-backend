from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
import os

load_dotenv()

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

chat = AzureChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name="gpt-4o",
    model_name="gpt-4o",
    api_version="2024-02-15-preview",
    streaming=True, 
    callbacks=[StreamingStdOutCallbackHandler()], 
    temperature=0.7,
    max_tokens=4096
)

print(chat([HumanMessage(content="Write me a song about sparkling water.")]))