from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from src.services.chat_service import ChatService

app = FastAPI(title="AI Chat API")
chat_service = ChatService()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "ai agent backend is running..."}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        messages = data.get("messages", [])
        agent_type = data.get("agent")
        
        if not messages:
            return JSONResponse(
                status_code=400,
                content={"error": "No messages provided"}
            )
            
        response_text = await chat_service.chat(messages, agent_type)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
