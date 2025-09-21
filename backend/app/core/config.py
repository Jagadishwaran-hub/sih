"""
Application configuration settings
"""

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # Database - Multiple connection strings for fallback
    DATABASE_URL: str = "mongodb://localhost:27017/railway_idss"
    DATABASE_URL_WITH_AUTH: str = "mongodb://railway_user:railway_pass@localhost:27017/railway_idss?authSource=admin"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Railway AI Decision Support System"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AI Engine
    AI_MODEL_PATH: Optional[str] = None
    OPTIMIZATION_TIMEOUT: int = 30  # seconds
    
    # Simulation
    SIMULATION_SPEED: float = 1.0
    MAX_SIMULATION_TIME: int = 3600  # seconds
    
    # Development mode
    DEV_MODE: bool = True
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()