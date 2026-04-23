import os
from dotenv import load_dotenv

# Load .env from parent directory first, then current directory
load_dotenv(dotenv_path="../.env")
load_dotenv()

class Config:
    port: int = int(os.getenv("PORT", 3001))
    db_path: str = os.getenv("DB_PATH", "./data/app.db")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    min_citation_score: float = float(os.getenv("MIN_CITATION_SCORE", 0.25))

config = Config()

if not config.openai_api_key:
    print("WARNING: OPENAI_API_KEY is not set. API calls will fail until configured.")