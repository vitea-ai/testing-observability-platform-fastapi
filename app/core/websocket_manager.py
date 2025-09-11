"""
WebSocket connection manager for real-time updates.
"""

import json
from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import asyncio
from app.core.logging import logger


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        # Map task_id to list of connected WebSockets
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Map WebSocket to set of task_ids it's subscribed to
        self.connection_subscriptions: Dict[WebSocket, Set[str]] = {}
        # Store last message for each task (for late joiners)
        self.last_messages: Dict[str, Dict[str, Any]] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, task_id: str) -> None:
        """
        Accept a new WebSocket connection and subscribe to a task.
        
        Args:
            websocket: The WebSocket connection
            task_id: The task ID to subscribe to
        """
        await websocket.accept()
        
        async with self._lock:
            # Add to task's connection list
            if task_id not in self.active_connections:
                self.active_connections[task_id] = []
            self.active_connections[task_id].append(websocket)
            
            # Track subscriptions for this connection
            if websocket not in self.connection_subscriptions:
                self.connection_subscriptions[websocket] = set()
            self.connection_subscriptions[websocket].add(task_id)
        
        logger.info(f"WebSocket connected for task {task_id}")
        
        # Send last message if available (for reconnections)
        if task_id in self.last_messages:
            try:
                await websocket.send_json(self.last_messages[task_id])
            except Exception as e:
                logger.warning(f"Failed to send last message to new connection: {e}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection and clean up subscriptions.
        
        Args:
            websocket: The WebSocket connection to remove
        """
        async with self._lock:
            # Get all task IDs this connection was subscribed to
            task_ids = self.connection_subscriptions.get(websocket, set())
            
            # Remove from each task's connection list
            for task_id in task_ids:
                if task_id in self.active_connections:
                    self.active_connections[task_id] = [
                        conn for conn in self.active_connections[task_id]
                        if conn != websocket
                    ]
                    # Clean up empty lists
                    if not self.active_connections[task_id]:
                        del self.active_connections[task_id]
            
            # Remove from subscription tracking
            if websocket in self.connection_subscriptions:
                del self.connection_subscriptions[websocket]
        
        logger.info(f"WebSocket disconnected from tasks: {task_ids}")
    
    async def subscribe(self, websocket: WebSocket, task_id: str) -> None:
        """
        Subscribe an existing connection to an additional task.
        
        Args:
            websocket: The WebSocket connection
            task_id: The task ID to subscribe to
        """
        async with self._lock:
            # Add to task's connection list
            if task_id not in self.active_connections:
                self.active_connections[task_id] = []
            if websocket not in self.active_connections[task_id]:
                self.active_connections[task_id].append(websocket)
            
            # Track subscription
            if websocket not in self.connection_subscriptions:
                self.connection_subscriptions[websocket] = set()
            self.connection_subscriptions[websocket].add(task_id)
        
        logger.debug(f"WebSocket subscribed to task {task_id}")
        
        # Send last message if available
        if task_id in self.last_messages:
            try:
                await websocket.send_json(self.last_messages[task_id])
            except Exception as e:
                logger.warning(f"Failed to send last message on subscribe: {e}")
    
    async def unsubscribe(self, websocket: WebSocket, task_id: str) -> None:
        """
        Unsubscribe a connection from a specific task.
        
        Args:
            websocket: The WebSocket connection
            task_id: The task ID to unsubscribe from
        """
        async with self._lock:
            # Remove from task's connection list
            if task_id in self.active_connections:
                self.active_connections[task_id] = [
                    conn for conn in self.active_connections[task_id]
                    if conn != websocket
                ]
                if not self.active_connections[task_id]:
                    del self.active_connections[task_id]
            
            # Update subscription tracking
            if websocket in self.connection_subscriptions:
                self.connection_subscriptions[websocket].discard(task_id)
        
        logger.debug(f"WebSocket unsubscribed from task {task_id}")
    
    async def broadcast(self, task_id: str, message: Dict[str, Any]) -> None:
        """
        Broadcast a message to all connections subscribed to a task.
        
        Args:
            task_id: The task ID to broadcast to
            message: The message to send
        """
        # Store last message for late joiners
        self.last_messages[task_id] = message
        
        # Get connections for this task
        connections = self.active_connections.get(task_id, [])
        
        if not connections:
            logger.debug(f"No active connections for task {task_id}")
            return
        
        # Send to all connections
        disconnected = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            await self.disconnect(conn)
        
        logger.debug(f"Broadcasted to {len(connections) - len(disconnected)} connections for task {task_id}")
    
    async def send_personal_message(
        self, 
        websocket: WebSocket, 
        message: Dict[str, Any]
    ) -> None:
        """
        Send a message to a specific WebSocket connection.
        
        Args:
            websocket: The WebSocket connection
            message: The message to send
        """
        try:
            await websocket.send_json(message)
        except WebSocketDisconnect:
            await self.disconnect(websocket)
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")
            await self.disconnect(websocket)
    
    def get_connection_count(self, task_id: str) -> int:
        """
        Get the number of active connections for a task.
        
        Args:
            task_id: The task ID
            
        Returns:
            Number of active connections
        """
        return len(self.active_connections.get(task_id, []))
    
    def get_all_task_ids(self) -> List[str]:
        """
        Get all task IDs with active connections.
        
        Returns:
            List of task IDs
        """
        return list(self.active_connections.keys())
    
    async def cleanup_completed_task(self, task_id: str, delay: int = 300) -> None:
        """
        Clean up resources for a completed task after a delay.
        
        Args:
            task_id: The task ID to clean up
            delay: Delay in seconds before cleanup (default 5 minutes)
        """
        await asyncio.sleep(delay)
        
        async with self._lock:
            # Remove last message
            if task_id in self.last_messages:
                del self.last_messages[task_id]
            
            # Disconnect any remaining connections
            if task_id in self.active_connections:
                connections = self.active_connections[task_id].copy()
                for conn in connections:
                    await self.disconnect(conn)
        
        logger.info(f"Cleaned up resources for task {task_id}")


# Global WebSocket manager instance
ws_manager = ConnectionManager()


class WebSocketManager:
    """
    Singleton WebSocket manager for the application.
    This class provides a simplified interface to the ConnectionManager.
    """
    
    _instance: Optional[ConnectionManager] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = ConnectionManager()
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> ConnectionManager:
        """Get the singleton ConnectionManager instance."""
        if cls._instance is None:
            cls._instance = ConnectionManager()
        return cls._instance
    
    async def connect(self, websocket: WebSocket, task_id: str) -> None:
        """Connect a WebSocket to a task."""
        await self._instance.connect(websocket, task_id)
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Disconnect a WebSocket."""
        await self._instance.disconnect(websocket)
    
    async def broadcast(self, task_id: str, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connections for a task."""
        await self._instance.broadcast(task_id, message)
    
    async def send_personal_message(
        self, 
        websocket: WebSocket, 
        message: Dict[str, Any]
    ) -> None:
        """Send a message to a specific connection."""
        await self._instance.send_personal_message(websocket, message)