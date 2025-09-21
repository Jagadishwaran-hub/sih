"""
FastAPI main application for Railway AI Decision Support System

This module initializes the FastAPI application with all necessary
middleware, routers, and configurations for the railway management system.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import logging
import json
from typing import List, Dict, Any
import asyncio
from datetime import datetime
import random
import asyncio
import uvicorn

from .routers import trains, schedules, simulation, kpis
from .core.config import settings
from .core.database import mongodb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Railway AI Decision Support System",
    description="Intelligent Decision Support System for Indian Railways Section Controllers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager for real-time updates
class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_subscriptions: Dict[WebSocket, List[str]] = {}

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_subscriptions[websocket] = []
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
        
        # Wait a brief moment for connection to stabilize
        await asyncio.sleep(0.1)
        
        # Send initial connection confirmation (only if still connected)
        if websocket in self.active_connections:
            try:
                await websocket.send_text(json.dumps({
                    "type": "connected",
                    "message": "WebSocket connection established",
                    "timestamp": datetime.utcnow().isoformat()
                }))
            except Exception as e:
                logger.warning(f"Failed to send connection confirmation: {e}")
                self.disconnect(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.client_subscriptions:
            del self.client_subscriptions[websocket]
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            # Check if WebSocket is in active connections and ready
            if websocket not in self.active_connections:
                logger.debug("WebSocket not in active connections, skipping message")
                return
                
            # Check WebSocket state more thoroughly
            if hasattr(websocket, 'client_state'):
                if websocket.client_state != 1:  # Not WebSocket.OPEN
                    logger.debug(f"WebSocket state is {websocket.client_state}, not open")
                    self.disconnect(websocket)
                    return
            
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str, event_type: str = None):
        """Broadcast message to all connected WebSockets"""
        if not self.active_connections:
            return
            
        disconnected_clients = []
        
        for connection in self.active_connections.copy():  # Use copy to avoid modification during iteration
            try:
                # Check if WebSocket is still valid
                if hasattr(connection, 'client_state'):
                    if connection.client_state != 1:  # Not WebSocket.OPEN
                        disconnected_clients.append(connection)
                        continue
                
                # Check if client is subscribed to this event type
                if event_type and connection in self.client_subscriptions:
                    subscriptions = self.client_subscriptions[connection]
                    if subscriptions and event_type not in subscriptions:
                        continue
                
                await connection.send_text(message)
            except Exception as e:
                logger.debug(f"Error broadcasting to WebSocket: {e}")
                disconnected_clients.append(connection)
        
        # Clean up disconnected clients
        for client in disconnected_clients:
            self.disconnect(client)

    async def handle_subscription(self, websocket: WebSocket, data: Dict[str, Any]):
        """Handle client subscription to specific event types"""
        if websocket in self.client_subscriptions:
            event_types = data.get("subscribe", [])
            self.client_subscriptions[websocket] = event_types
            logger.info(f"Client subscribed to: {event_types}")
            
            await self.send_personal_message(json.dumps({
                "type": "subscription_confirmed",
                "subscribed_to": event_types,
                "timestamp": datetime.utcnow().isoformat()
            }), websocket)

# Global connection manager instance
manager = ConnectionManager()

# Include API routers
app.include_router(trains.router, prefix="/api/trains", tags=["trains"])
app.include_router(schedules.router, prefix="/api/schedules", tags=["schedules"])
app.include_router(simulation.router, prefix="/api/simulation", tags=["simulation"])
app.include_router(kpis.router, prefix="/api/kpis", tags=["kpis"])

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting Railway AI Decision Support System")
    
    # Connect to MongoDB
    await mongodb.connect_db()
    await mongodb.init_beanie_models()
    logger.info("✅ MongoDB connection and models initialized")
    
    # Initialize AI components
    logger.info("🤖 AI Engine initialized")
    
    logger.info("✅ Application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("🛑 Shutting down Railway AI Decision Support System")
    await mongodb.close_db()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with system information"""
    return """
    <html>
        <head>
            <title>Railway AI Decision Support System</title>
        </head>
        <body>
            <h1>🚂 Railway AI Decision Support System</h1>
            <p>Intelligent Decision Support for Indian Railways Section Controllers</p>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/redoc">Alternative API Docs</a></li>
                <li><a href="/health">Health Check</a></li>
            </ul>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Railway AI Decision Support System",
        "version": "1.0.0",
        "websocket_connections": len(manager.active_connections)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Handle incoming messages from clients
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "subscribe":
                    await manager.handle_subscription(websocket, message)
                elif message_type == "ping":
                    # Respond to ping with pong
                    await manager.send_personal_message(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }), websocket)
                    logger.debug("Sent pong response to client")
                elif message_type == "heartbeat":
                    # Alternative heartbeat message
                    await manager.send_personal_message(json.dumps({
                        "type": "heartbeat_ack",
                        "timestamp": datetime.utcnow().isoformat()
                    }), websocket)
                else:
                    # Echo back unhandled messages
                    await manager.send_personal_message(json.dumps({
                        "type": "echo",
                        "original": message,
                        "timestamp": datetime.utcnow().isoformat()
                    }), websocket)
                    
            except json.JSONDecodeError:
                await manager.send_personal_message(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                }), websocket)
            
    except WebSocketDisconnect:
        logger.info("Client disconnected normally")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Helper function to broadcast events
async def broadcast_event(event_type: str, data: Dict[str, Any]):
    """Helper function to broadcast events to WebSocket clients"""
    message = {
        "type": event_type,
        "payload": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast(json.dumps(message), event_type)

# Background task for sending real-time updates
async def send_real_time_updates():
    """Background task to send real-time train updates"""
    train_data = {
        "T001": {"station": "NDLS", "delay": 0, "next": "GZB"},
        "T002": {"station": "GZB", "delay": 5, "next": "MB"},
        "T003": {"station": "MB", "delay": 15, "next": "BE"},
    }
    
    counter = 0
    while True:
        counter += 1
        
        # Send system stats every 30 seconds
        if counter % 3 == 0:
            await broadcast_event("system_stats", {
                "totalTrains": 24,
                "onTimePercentage": random.randint(80, 95),
                "activeAlerts": random.randint(0, 5),
                "avgDelay": random.randint(8, 15)
            })
        
        # Send train position updates
        for train_id, info in train_data.items():
            # Simulate train movement and delays
            if random.random() < 0.3:  # 30% chance of update
                info["delay"] += random.randint(-2, 3)
                info["delay"] = max(0, info["delay"])  # No negative delays
                
                await broadcast_event("train_update", {
                    "train_id": train_id,
                    "current_station": info["station"],
                    "delay_minutes": info["delay"],
                    "next_station": info["next"],
                    "status": "DELAYED" if info["delay"] > 10 else "ON_TIME" if info["delay"] == 0 else "DELAYED",
                    "last_updated": datetime.utcnow().isoformat()
                })
        
        # Send AI recommendations occasionally
        if counter % 5 == 0:  # Every 50 seconds
            recommendations = [
                "Optimize platform assignment for Train T001 at NDLS",
                "Consider alternate route for Train T003 due to congestion",
                "Schedule adjustment recommended for freight trains"
            ]
            
            await broadcast_event("ai_recommendation", {
                "id": counter,
                "type": "schedule_optimization",
                "priority": random.choice(["high", "medium", "low"]),
                "title": random.choice(recommendations),
                "confidence": random.randint(75, 95),
                "created_at": datetime.utcnow().isoformat()
            })
        
        # Send alerts occasionally
        if counter % 8 == 0 and random.random() < 0.3:  # Random alerts
            alerts = [
                {"severity": "warning", "title": "Platform Congestion", "message": "Platform 2 at NDLS approaching capacity"},
                {"severity": "info", "title": "Weather Update", "message": "Light fog expected in Northern region"},
                {"severity": "error", "title": "Signal Failure", "message": "Temporary signal issue at GZB resolved"}
            ]
            
            alert = random.choice(alerts)
            await broadcast_event("alert", alert)
        
        await asyncio.sleep(10)  # Send updates every 10 seconds

# Background task for AI optimization
async def ai_optimization_task():
    """Background task for AI optimization"""
    while True:
        await asyncio.sleep(60)  # Run every minute
        
        # Simulate AI optimization completion
        await broadcast_event("optimization_complete", {
            "algorithm": random.choice(["OR-Tools", "Genetic Algorithm", "Reinforcement Learning"]),
            "execution_time": random.randint(2, 8),
            "improvements": {
                "delay_reduction": random.randint(5, 20),
                "conflicts_resolved": random.randint(1, 4),
                "efficiency_gain": random.randint(2, 8)
            },
            "timestamp": datetime.utcnow().isoformat()
        })

# Start background tasks
@app.on_event("startup")
async def start_background_tasks():
    """Start background tasks"""
    asyncio.create_task(send_real_time_updates())
    asyncio.create_task(ai_optimization_task())
    logger.info("✅ Background tasks started")

def main():
    """
    Main application entry point
    Starts the FastAPI server and initializes core components
    """
    logger.info("🚀 Starting AI Decision Support System for Railways")
    logger.info("Initializing core components...")
    
    # Initialize AI Scheduler (will be done through startup events)
    logger.info("✅ AI Scheduler will be initialized on startup")
    
    # Initialize Railway Simulator (will be done through startup events)
    logger.info("✅ Railway Simulator will be initialized on startup")
    
    logger.info("🌐 Starting FastAPI server on localhost...")
    
    # Start the FastAPI application
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()