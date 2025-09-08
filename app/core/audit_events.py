"""
Audit event emitter for business events using HTTP OTLP.

This module provides reliable emission of business audit events to the Vitea
observability stack using direct HTTP OTLP protocol (recommended approach).
Events are sent to ClickHouse via the OpenTelemetry collector.
"""

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from app.core.logging import logger


class AuditEventEmitter:
    """
    Emits business audit events via HTTP OTLP protocol.
    
    This implementation uses direct HTTP OTLP which is more reliable than
    the Python SDK approach due to known SDK issues with log record handling.
    """
    
    def __init__(
        self, 
        service_name: str, 
        service_version: str = "1.0.0",
        environment: Optional[str] = None
    ):
        """
        Initialize the audit event emitter.
        
        Args:
            service_name: Name of the service emitting events
            service_version: Version of the service
            environment: Deployment environment (defaults to APP_ENV)
        """
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment or os.getenv("APP_ENV", "development")
        
        # Use localhost for local development, host.docker.internal for Docker
        default_endpoint = "http://localhost:10318/v1/logs"
        if os.path.exists("/.dockerenv"):
            default_endpoint = "http://host.docker.internal:10318/v1/logs"
            
        self.otlp_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
            default_endpoint
        )

    def _send_otlp_event(self, attributes: List[Dict[str, Any]], body: str) -> None:
        """
        Send an event via HTTP OTLP protocol.
        
        Args:
            attributes: List of key-value attributes for the event
            body: Human-readable event description
        """
        timestamp_ns = int(time.time() * 1_000_000_000)
        
        otlp_payload = {
            "resourceLogs": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": self.service_name}},
                        {"key": "service.version", "value": {"stringValue": self.service_version}},
                        {"key": "deployment.environment", "value": {"stringValue": self.environment}}
                    ]
                },
                "scopeLogs": [{
                    "scope": {"name": "business_audit", "version": "1.0.0"},
                    "logRecords": [{
                        "timeUnixNano": str(timestamp_ns),
                        "severityText": "INFO",
                        "severityNumber": 9,
                        "body": {"stringValue": body},
                        "attributes": attributes
                    }]
                }]
            }]
        }
        
        try:
            response = requests.post(
                self.otlp_endpoint,
                json=otlp_payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(
                    "Failed to send audit event",
                    service_name=self.service_name,
                    status_code=response.status_code,
                    response=response.text[:500]
                )
        except Exception as e:
            logger.error(
                "Error sending audit event",
                service_name=self.service_name,
                error=str(e)
            )

    def emit_dataset_upload(
        self,
        *,
        dataset_id: str,
        dataset_name: str,
        user_id: str,
        file_size: int,
        row_count: int,
        conversation_count: int,
        format_type: str = "csv",
        processing_time_ms: int,
        status: str = "success"
    ) -> None:
        """
        Emit an audit event for dataset upload.
        
        Args:
            dataset_id: Unique identifier for the dataset
            dataset_name: Name of the dataset
            user_id: User who uploaded the dataset
            file_size: Size of the uploaded file in bytes
            row_count: Number of rows in the dataset
            conversation_count: Number of conversations extracted
            format_type: Format of the uploaded file
            processing_time_ms: Time taken to process the upload
            status: Upload status (success/failed)
        """
        correlation_id = f"dataset-{dataset_id}-{int(time.time())}"
        
        attributes = [
            {"key": "event.kind", "value": {"stringValue": "audit"}},
            {"key": "event.name", "value": {"stringValue": "dataset_upload"}},
            {"key": "dataset_id", "value": {"stringValue": dataset_id}},
            {"key": "dataset_name", "value": {"stringValue": dataset_name}},
            {"key": "user_id", "value": {"stringValue": user_id}},
            {"key": "file_size", "value": {"stringValue": str(file_size)}},
            {"key": "row_count", "value": {"stringValue": str(row_count)}},
            {"key": "conversation_count", "value": {"stringValue": str(conversation_count)}},
            {"key": "format_type", "value": {"stringValue": format_type}},
            {"key": "processing_time_ms", "value": {"stringValue": str(processing_time_ms)}},
            {"key": "status", "value": {"stringValue": status}},
            {"key": "correlation_id", "value": {"stringValue": correlation_id}}
        ]
        
        body = f"Dataset upload: {dataset_name} ({row_count} rows, {conversation_count} conversations) - {status}"
        self._send_otlp_event(attributes, body)

    def emit_experiment_execution(
        self,
        *,
        experiment_id: str,
        experiment_name: str,
        dataset_id: str,
        model_name: str,
        user_id: str,
        test_count: int,
        success_count: int,
        failure_count: int,
        avg_latency_ms: float,
        total_tokens: int,
        execution_time_ms: int,
        status: str = "completed"
    ) -> None:
        """
        Emit an audit event for experiment execution.
        
        Args:
            experiment_id: Unique identifier for the experiment
            experiment_name: Name of the experiment
            dataset_id: Dataset used for the experiment
            model_name: AI model used
            user_id: User who ran the experiment
            test_count: Total number of tests executed
            success_count: Number of successful tests
            failure_count: Number of failed tests
            avg_latency_ms: Average latency per test
            total_tokens: Total tokens consumed
            execution_time_ms: Total execution time
            status: Execution status
        """
        correlation_id = f"experiment-{experiment_id}-{int(time.time())}"
        
        attributes = [
            {"key": "event.kind", "value": {"stringValue": "audit"}},
            {"key": "event.name", "value": {"stringValue": "experiment_execution"}},
            {"key": "experiment_id", "value": {"stringValue": experiment_id}},
            {"key": "experiment_name", "value": {"stringValue": experiment_name}},
            {"key": "dataset_id", "value": {"stringValue": dataset_id}},
            {"key": "model_name", "value": {"stringValue": model_name}},
            {"key": "user_id", "value": {"stringValue": user_id}},
            {"key": "test_count", "value": {"stringValue": str(test_count)}},
            {"key": "success_count", "value": {"stringValue": str(success_count)}},
            {"key": "failure_count", "value": {"stringValue": str(failure_count)}},
            {"key": "avg_latency_ms", "value": {"stringValue": str(avg_latency_ms)}},
            {"key": "total_tokens", "value": {"stringValue": str(total_tokens)}},
            {"key": "execution_time_ms", "value": {"stringValue": str(execution_time_ms)}},
            {"key": "status", "value": {"stringValue": status}},
            {"key": "correlation_id", "value": {"stringValue": correlation_id}}
        ]
        
        body = f"Experiment execution: {experiment_name} ({test_count} tests, {success_count}/{failure_count} success/fail) - {status}"
        self._send_otlp_event(attributes, body)

    def emit_evaluation_result(
        self,
        *,
        evaluation_id: str,
        experiment_id: str,
        test_id: str,
        evaluator_name: str,
        score: float,
        passed: bool,
        latency_ms: int,
        tokens_used: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Emit an audit event for evaluation result.
        
        Args:
            evaluation_id: Unique identifier for the evaluation
            experiment_id: Parent experiment ID
            test_id: Test case ID
            evaluator_name: Name of the evaluator used
            score: Evaluation score
            passed: Whether the evaluation passed
            latency_ms: Evaluation latency
            tokens_used: Tokens consumed
            metadata: Additional metadata
        """
        correlation_id = f"eval-{evaluation_id}-{int(time.time())}"
        
        attributes = [
            {"key": "event.kind", "value": {"stringValue": "audit"}},
            {"key": "event.name", "value": {"stringValue": "evaluation_result"}},
            {"key": "evaluation_id", "value": {"stringValue": evaluation_id}},
            {"key": "experiment_id", "value": {"stringValue": experiment_id}},
            {"key": "test_id", "value": {"stringValue": test_id}},
            {"key": "evaluator_name", "value": {"stringValue": evaluator_name}},
            {"key": "score", "value": {"stringValue": str(score)}},
            {"key": "passed", "value": {"stringValue": str(passed).lower()}},
            {"key": "latency_ms", "value": {"stringValue": str(latency_ms)}},
            {"key": "tokens_used", "value": {"stringValue": str(tokens_used)}},
            {"key": "correlation_id", "value": {"stringValue": correlation_id}}
        ]
        
        # Add metadata if provided
        if metadata:
            metadata_hash = hashlib.md5(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
            attributes.append({"key": "metadata_hash", "value": {"stringValue": metadata_hash}})
        
        body = f"Evaluation: {evaluator_name} scored {score:.2f} ({'passed' if passed else 'failed'})"
        self._send_otlp_event(attributes, body)

    def emit_api_call(
        self,
        *,
        endpoint: str,
        method: str,
        user_id: Optional[str] = None,
        status_code: int,
        latency_ms: int,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Emit an audit event for API calls.
        
        Args:
            endpoint: API endpoint called
            method: HTTP method
            user_id: User making the request
            status_code: HTTP response status code
            latency_ms: Request latency
            request_size: Size of request body
            response_size: Size of response body
            error_message: Error message if failed
        """
        correlation_id = f"api-{method}-{int(time.time())}-{hash(endpoint) % 10000}"
        
        attributes = [
            {"key": "event.kind", "value": {"stringValue": "audit"}},
            {"key": "event.name", "value": {"stringValue": "api_call"}},
            {"key": "endpoint", "value": {"stringValue": endpoint}},
            {"key": "method", "value": {"stringValue": method}},
            {"key": "status_code", "value": {"stringValue": str(status_code)}},
            {"key": "latency_ms", "value": {"stringValue": str(latency_ms)}},
            {"key": "correlation_id", "value": {"stringValue": correlation_id}}
        ]
        
        if user_id:
            attributes.append({"key": "user_id", "value": {"stringValue": user_id}})
        if request_size is not None:
            attributes.append({"key": "request_size", "value": {"stringValue": str(request_size)}})
        if response_size is not None:
            attributes.append({"key": "response_size", "value": {"stringValue": str(response_size)}})
        if error_message:
            attributes.append({"key": "error_message", "value": {"stringValue": error_message[:500]}})
        
        body = f"API call: {method} {endpoint} -> {status_code}"
        self._send_otlp_event(attributes, body)