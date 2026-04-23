from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import config
from app.routers import documents_router, questions_router
from app import schemas

app = FastAPI(
    title="Document Q&A API",
    description="API for document ingestion and question answering with citations",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents_router)
app.include_router(questions_router)

@app.get("/health", response_model=schemas.HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=config.port,
        reload=True
    )