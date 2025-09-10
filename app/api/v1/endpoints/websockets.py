"""
WebSocket endpoints for real-time updates.
"""

import asyncio
from typing import Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.responses import HTMLResponse
from app.core.logging import logger
from app.core.websocket_manager import ws_manager
from app.core.dependencies import get_current_user_optional
from app.services.queue_service import QueueService
from celery.result import AsyncResult
from app.workers.celery_app import celery_app

router = APIRouter()


@router.websocket("/evaluations/{task_id}")
async def websocket_evaluation_updates(
    websocket: WebSocket,
    task_id: str,
):
    """
    WebSocket endpoint for real-time evaluation updates.
    
    Clients connect to this endpoint to receive real-time updates about
    their evaluation tasks.
    
    Protocol:
    - Connect: ws://host/ws/evaluations/{task_id}
    - Receive: JSON messages with status updates
    - Send: JSON commands (subscribe, unsubscribe, ping)
    """
    await ws_manager.connect(websocket, task_id)
    logger.info(f"WebSocket connection established for task {task_id}")
    
    try:
        # Send initial status
        queue_service = QueueService()
        status = await queue_service.get_task_status(task_id)
        await websocket.send_json({
            "type": "status",
            "task_id": task_id,
            "status": status.get("status", "unknown"),
            "message": "Connected to evaluation updates"
        })
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages with timeout for keep-alive
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                # Handle different message types
                message_type = data.get("type")
                
                if message_type == "ping":
                    # Respond to ping with pong
                    await websocket.send_json({"type": "pong"})
                    
                elif message_type == "subscribe":
                    # Subscribe to additional task
                    additional_task_id = data.get("task_id")
                    if additional_task_id:
                        await ws_manager._instance.subscribe(websocket, additional_task_id)
                        await websocket.send_json({
                            "type": "subscribed",
                            "task_id": additional_task_id
                        })
                        
                elif message_type == "unsubscribe":
                    # Unsubscribe from a task
                    unsubscribe_task_id = data.get("task_id")
                    if unsubscribe_task_id:
                        await ws_manager._instance.unsubscribe(websocket, unsubscribe_task_id)
                        await websocket.send_json({
                            "type": "unsubscribed",
                            "task_id": unsubscribe_task_id
                        })
                        
                elif message_type == "status":
                    # Request current status
                    status = await queue_service.get_task_status(task_id)
                    await websocket.send_json({
                        "type": "status",
                        "task_id": task_id,
                        **status
                    })
                    
            except asyncio.TimeoutError:
                # Send keep-alive ping
                try:
                    await websocket.send_json({"type": "ping"})
                except:
                    break
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}")
    finally:
        await ws_manager.disconnect(websocket)


@router.websocket("/experiments/{experiment_id}/live")
async def websocket_experiment_live(
    websocket: WebSocket,
    experiment_id: str,
):
    """
    WebSocket endpoint for live experiment updates.
    
    This endpoint allows clients to monitor all evaluations
    for a specific experiment in real-time.
    """
    await websocket.accept()
    logger.info(f"WebSocket connection for experiment {experiment_id} monitoring")
    
    subscribed_tasks = set()
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connected",
            "experiment_id": experiment_id,
            "message": "Connected to experiment live updates"
        })
        
        while True:
            try:
                # Wait for client messages
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                message_type = data.get("type")
                
                if message_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    
                elif message_type == "track_task":
                    # Start tracking a new evaluation task for this experiment
                    task_id = data.get("task_id")
                    if task_id and task_id not in subscribed_tasks:
                        await ws_manager._instance.subscribe(websocket, task_id)
                        subscribed_tasks.add(task_id)
                        await websocket.send_json({
                            "type": "tracking",
                            "task_id": task_id,
                            "experiment_id": experiment_id
                        })
                        
            except asyncio.TimeoutError:
                # Send keep-alive
                try:
                    await websocket.send_json({"type": "ping"})
                except:
                    break
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for experiment {experiment_id}")
    except Exception as e:
        logger.error(f"WebSocket error for experiment {experiment_id}: {e}")
    finally:
        # Unsubscribe from all tasks
        for task_id in subscribed_tasks:
            await ws_manager._instance.unsubscribe(websocket, task_id)
        await websocket.close()


@router.get("/test")
async def websocket_test_page():
    """
    Simple HTML page for testing WebSocket connections.
    Only available in development mode.
    """
    from app.core.config import settings
    
    if settings.deployment_tier != "development":
        return {"error": "Test page only available in development mode"}
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebSocket Test</title>
    </head>
    <body>
        <h1>WebSocket Evaluation Updates Test</h1>
        <form>
            <input type="text" id="taskId" placeholder="Enter Task ID" />
            <button type="button" onclick="connect()">Connect</button>
            <button type="button" onclick="disconnect()">Disconnect</button>
        </form>
        <div id="messages"></div>
        
        <script>
            let ws = null;
            
            function connect() {
                const taskId = document.getElementById('taskId').value;
                if (!taskId) {
                    alert('Please enter a task ID');
                    return;
                }
                
                ws = new WebSocket(`ws://localhost:8000/ws/evaluations/${taskId}`);
                
                ws.onopen = () => {
                    console.log('Connected');
                    addMessage('Connected to task: ' + taskId);
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    console.log('Message:', data);
                    addMessage(JSON.stringify(data, null, 2));
                };
                
                ws.onerror = (error) => {
                    console.error('Error:', error);
                    addMessage('Error: ' + error);
                };
                
                ws.onclose = () => {
                    console.log('Disconnected');
                    addMessage('Disconnected');
                };
            }
            
            function disconnect() {
                if (ws) {
                    ws.close();
                }
            }
            
            function addMessage(message) {
                const messages = document.getElementById('messages');
                const messageElement = document.createElement('pre');
                messageElement.textContent = new Date().toISOString() + ': ' + message;
                messages.appendChild(messageElement);
            }
            
            // Send ping every 20 seconds to keep connection alive
            setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'ping'}));
                }
            }, 20000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)