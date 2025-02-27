from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# from src.models.gpt4o import GitHubGPTAgent
from src.api.endpoints import chat
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Agent API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming {request.method} request to {request.url.path}")
    response = await call_next(request)
    logger.info(f"Returning response with status code {response.status_code}")
    return response

#include router
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])

# class WeatherRequest(BaseModel):
#     city: str
#     units: str = "metric"

# class GPTRequest(BaseModel):
#     query: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "running", "message": "AI Agent is running", "environment": "development"}


# @app.post("/gpt")
# async def gpt_endpoint(request: GPTRequest):
#     try:
#         agent = GitHubGPTAgent()
#         response = await agent.run(request.query)
#         return {"data": response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
