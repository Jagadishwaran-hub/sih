"""
Train management API endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta
import logging
from beanie import PydanticObjectId

try:
    from ..models.database import Train, ScheduleEntry
except ImportError:
    # Handle case where beanie is not available
    Train = None
    ScheduleEntry = None
    PydanticObjectId = str

from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

class TrainCreate(BaseModel):
    train_number: str
    train_name: str
    train_type: str  # EXPRESS, PASSENGER, FREIGHT
    priority: int = 1
    source_station: str
    destination_station: str
    departure_time: datetime
    arrival_time: datetime
    current_delay: int = 0
    current_station: Optional[str] = None

class TrainUpdate(BaseModel):
    train_name: Optional[str] = None
    train_type: Optional[str] = None
    priority: Optional[int] = None
    current_delay: Optional[int] = None
    current_station: Optional[str] = None
    is_active: Optional[bool] = None

class TrainResponse(BaseModel):
    id: str
    train_number: str
    train_name: str
    train_type: str
    priority: int
    source_station: str
    destination_station: str
    departure_time: datetime
    arrival_time: datetime
    current_delay: int
    current_station: Optional[str] = None
    is_active: bool
    created_at: datetime

@router.get("/", response_model=List[TrainResponse])
async def get_trains(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    train_type: Optional[str] = None,
    is_active: Optional[bool] = True
):
    """Get all trains with optional filtering"""
    try:
        # Build query
        query = {}
        if train_type:
            query["train_type"] = train_type
        if is_active is not None:
            query["is_active"] = is_active
        
        # Execute query
        trains = await Train.find(query).skip(skip).limit(limit).to_list()
        
        # Convert to response format
        train_responses = []
        for train in trains:
            train_responses.append(TrainResponse(
                id=str(train.id),
                train_number=train.train_number,
                train_name=train.train_name,
                train_type=train.train_type,
                priority=train.priority,
                source_station=train.source_station,
                destination_station=train.destination_station,
                departure_time=train.departure_time,
                arrival_time=train.arrival_time,
                current_delay=train.current_delay,
                current_station=train.current_station,
                is_active=train.is_active,
                created_at=train.created_at
            ))
        
        return train_responses
        
    except Exception as e:
        logger.error(f"Failed to get trains: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trains: {str(e)}")

@router.get("/{train_id}", response_model=TrainResponse)
async def get_train(train_id: str):
    """Get a specific train by ID"""
    try:
        if not PydanticObjectId.is_valid(train_id):
            raise HTTPException(status_code=400, detail="Invalid train ID format")
        
        train = await Train.get(PydanticObjectId(train_id))
        if not train:
            raise HTTPException(status_code=404, detail="Train not found")
        
        return TrainResponse(
            id=str(train.id),
            train_number=train.train_number,
            train_name=train.train_name,
            train_type=train.train_type,
            priority=train.priority,
            source_station=train.source_station,
            destination_station=train.destination_station,
            departure_time=train.departure_time,
            arrival_time=train.arrival_time,
            current_delay=train.current_delay,
            current_station=train.current_station,
            is_active=train.is_active,
            created_at=train.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get train {train_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get train: {str(e)}")

@router.post("/", response_model=TrainResponse)
async def create_train(train_data: TrainCreate):
    """Create a new train"""
    try:
        # Check if train number already exists
        existing_train = await Train.find_one({"train_number": train_data.train_number})
        if existing_train:
            raise HTTPException(status_code=400, detail="Train number already exists")
        
        # Create new train
        train = Train(
            train_number=train_data.train_number,
            train_name=train_data.train_name,
            train_type=train_data.train_type,
            priority=train_data.priority,
            source_station=train_data.source_station,
            destination_station=train_data.destination_station,
            departure_time=train_data.departure_time,
            arrival_time=train_data.arrival_time,
            current_delay=train_data.current_delay,
            current_station=train_data.current_station
        )
        
        await train.save()
        
        logger.info(f"Created new train: {train.train_number}")
        
        return TrainResponse(
            id=str(train.id),
            train_number=train.train_number,
            train_name=train.train_name,
            train_type=train.train_type,
            priority=train.priority,
            source_station=train.source_station,
            destination_station=train.destination_station,
            departure_time=train.departure_time,
            arrival_time=train.arrival_time,
            current_delay=train.current_delay,
            current_station=train.current_station,
            is_active=train.is_active,
            created_at=train.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create train: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create train: {str(e)}")

@router.get("/active/current")
async def get_active_trains():
    """Get all currently active trains"""
    try:
        active_trains = await Train.find({"is_active": True}).to_list()
        
        # Add current position and delay information
        train_data = []
        for train in active_trains:
            train_info = {
                "id": str(train.id),
                "train_number": train.train_number,
                "train_name": train.train_name,
                "train_type": train.train_type,
                "priority": train.priority,
                "current_delay": train.current_delay,
                "current_station": train.current_station,
                "source_station": train.source_station,
                "destination_station": train.destination_station,
                "status": "ON_TIME" if train.current_delay <= 5 else "DELAYED"
            }
            train_data.append(train_info)
        
        return {
            "total_active_trains": len(train_data),
            "trains": train_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get active trains: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get active trains: {str(e)}")

@router.put("/{train_id}/delay")
async def update_train_delay(
    train_id: str,
    delay_minutes: int,
    current_station: str = None
):
    """Update train delay and current position"""
    try:
        if not PydanticObjectId.is_valid(train_id):
            raise HTTPException(status_code=400, detail="Invalid train ID format")
        
        train = await Train.get(PydanticObjectId(train_id))
        if not train:
            raise HTTPException(status_code=404, detail="Train not found")
        
        old_delay = train.current_delay
        train.current_delay = delay_minutes
        if current_station:
            train.current_station = current_station
        
        await train.save()
        logger.info(f"Updated train {train.train_number} delay to {delay_minutes} minutes")
        
        return {
            "message": "Train delay updated successfully",
            "train_number": train.train_number,
            "old_delay": old_delay,
            "new_delay": delay_minutes,
            "current_station": train.current_station
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update train delay: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update train delay: {str(e)}")

@router.get("/delayed/summary")
async def get_delayed_trains_summary():
    """Get summary of delayed trains"""
    try:
        delayed_trains = await Train.find({"current_delay": {"$gt": 5}, "is_active": True}).to_list()
        
        if not delayed_trains:
            return {
                "total_delayed": 0,
                "avg_delay": 0,
                "max_delay": 0,
                "delayed_trains": []
            }
        
        total_delay = sum(t.current_delay for t in delayed_trains)
        avg_delay = total_delay / len(delayed_trains)
        max_delay = max(t.current_delay for t in delayed_trains)
        
        summary = {
            "total_delayed": len(delayed_trains),
            "avg_delay": round(avg_delay, 2),
            "max_delay": max_delay,
            "delayed_trains": [
                {
                    "id": str(t.id),
                    "train_number": t.train_number,
                    "train_name": t.train_name,
                    "delay_minutes": t.current_delay,
                    "current_station": t.current_station
                }
                for t in delayed_trains
            ]
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get delayed trains summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get delayed trains summary: {str(e)}")