from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from src.api.endpoints import chat
from fastapi.responses import FileResponse

load_dotenv()

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "AI Agent is running...",
        "environment": "development",
        "version": "1.0.0"
    }

@app.get("/test")
def index():
    return FileResponse("index.html")

# Include chat router
app.include_router(chat.router, prefix="/api/v1")