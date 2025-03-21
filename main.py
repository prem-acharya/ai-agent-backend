from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from src.api.endpoints import chat
from fastapi.responses import FileResponse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

load_dotenv()
logger.info("Environment variables loaded")

app = FastAPI()
logger.info("FastAPI application initialized")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured")

@app.get("/")
def root():
    logger.info("Root endpoint called")
    return {
        "status": "running",
        "message": "AI Agent is running...",
        "environment": "development",
        "version": "1.0.0"
    }

@app.get("/test")
def index():
    logger.info("Test endpoint called")
    return FileResponse("index.html")

# Include chat router
app.include_router(chat.router, prefix="/api/v1")
logger.info("Chat router included with prefix /api/v1")