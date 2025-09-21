"""
MongoDB database configuration and connection management
"""

import logging
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from fastapi import HTTPException
from .config import settings

logger = logging.getLogger(__name__)

class MongoDB:
    """MongoDB connection manager"""
    
    def __init__(self):
        self.client: AsyncIOMotorClient = None
        self.database = None
    
    async def connect_db(self):
        """Create database connection with fallback options"""
        connection_strings = [
            settings.DATABASE_URL,  # Try without auth first
            settings.DATABASE_URL_WITH_AUTH,  # Try with auth
            "mongodb://localhost:27017/railway_idss"  # Final fallback
        ]
        
        last_error = None
        
        for i, conn_str in enumerate(connection_strings):
            try:
                logger.info(f"🔄 Attempting MongoDB connection {i+1}/{len(connection_strings)}")
                self.client = AsyncIOMotorClient(conn_str)
                
                # Extract database name from URL
                db_name = conn_str.split('/')[-1].split('?')[0] or "railway_idss"
                self.database = self.client[db_name]
                
                # Test connection
                await self.client.admin.command('ping')
                logger.info(f"✅ Connected to MongoDB database: {db_name}")
                return  # Success - exit function
                
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Connection attempt {i+1} failed: {e}")
                if self.client:
                    self.client.close()
                    self.client = None
                continue
        
        # If we get here, all connection attempts failed
        if settings.DEV_MODE:
            logger.warning("🔧 All MongoDB connections failed - starting in offline mode")
            logger.warning("📋 To run with database:")
            logger.warning("   1. Install MongoDB: https://www.mongodb.com/try/download/community")
            logger.warning("   2. Start MongoDB service")
            logger.warning("   3. Restart the application")
            return  # Continue without database in dev mode
        else:
            logger.error(f"❌ All MongoDB connection attempts failed. Last error: {last_error}")
            raise last_error
    
    async def close_db(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    async def init_beanie_models(self):
        """Initialize Beanie ODM with models"""
        if self.database is None:
            logger.warning("⚠️ No database connection - skipping Beanie initialization")
            return
            
        try:
            from ..models.database import Train, Station, ScheduleEntry, Conflict, KPILog, AIDecision
            
            await init_beanie(
                database=self.database,
                document_models=[
                    Train,
                    Station, 
                    ScheduleEntry,
                    Conflict,
                    KPILog,
                    AIDecision
                ]
            )
            logger.info("✅ Beanie ODM initialized with models")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Beanie: {e}")
            if not settings.DEV_MODE:
                raise e

# Global MongoDB instance
mongodb = MongoDB()

# Dependency to get database
async def get_database():
    """Get database instance"""
    if mongodb.database is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return mongodb.database