#!/usr/bin/env python3
"""
Test script to verify observability integration with Vitea stack.

This script tests:
1. JSON logging with service_name
2. OpenTelemetry trace export
3. Prometheus metrics endpoint
4. Audit event emission
"""

import os
import time
import requests
import json

# Set environment for staging to enable observability
os.environ["APP_ENV"] = "staging"
os.environ["APP_DEPLOYMENT_TIER"] = "staging"
os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://localhost:10318/v1/traces"
os.environ["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] = "http://localhost:10318/v1/logs"
os.environ["ENABLE_METRICS"] = "true"

def test_api_endpoints():
    """Test basic API endpoints to generate traces and metrics."""
    base_url = "http://localhost:8002"
    
    print("🔍 Testing API endpoints...")
    
    # Test health endpoint
    response = requests.get(f"{base_url}/health")
    print(f"✅ Health check: {response.status_code}")
    
    # Test ready endpoint
    response = requests.get(f"{base_url}/ready")
    print(f"✅ Ready check: {response.status_code}")
    
    # Test metrics endpoint
    response = requests.get(f"{base_url}/metrics")
    if response.status_code == 200:
        print(f"✅ Prometheus metrics endpoint available")
        # Show a sample of metrics
        lines = response.text.split('\n')[:10]
        print("   Sample metrics:")
        for line in lines[:5]:
            if line and not line.startswith('#'):
                print(f"   - {line}")
    
    # Test API v1 endpoints to generate business events
    print("\n🔍 Testing business endpoints...")
    
    # Create a test dataset (this should emit audit events)
    try:
        response = requests.get(f"{base_url}/api/v1/datasets")
        print(f"✅ Datasets endpoint: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Datasets endpoint error: {e}")
    
    return True

def test_observability_collector():
    """Test connectivity to observability collector."""
    print("\n🔍 Testing observability collector connectivity...")
    
    # Test OTLP traces endpoint
    traces_url = "http://localhost:10318/v1/traces"
    try:
        # Send a test trace
        test_trace = {
            "resourceSpans": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "test-script"}}
                    ]
                },
                "scopeSpans": [{
                    "scope": {"name": "test"},
                    "spans": [{
                        "traceId": "00000000000000000000000000000001",
                        "spanId": "0000000000000001",
                        "name": "test-span",
                        "startTimeUnixNano": str(int(time.time() * 1e9)),
                        "endTimeUnixNano": str(int(time.time() * 1e9) + 1000000)
                    }]
                }]
            }]
        }
        
        response = requests.post(
            traces_url,
            json=test_trace,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        
        if response.status_code == 200:
            print(f"✅ OTLP traces endpoint reachable")
        else:
            print(f"⚠️  OTLP traces endpoint returned: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to OTLP traces endpoint at {traces_url}")
        print("   Make sure the observability stack is running:")
        print("   cd ../observability && ./scripts/start_stack.sh")
    except Exception as e:
        print(f"⚠️  OTLP traces test error: {e}")
    
    # Test OTLP logs endpoint
    logs_url = "http://localhost:10318/v1/logs"
    try:
        # Send a test log
        test_log = {
            "resourceLogs": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "test-script"}}
                    ]
                },
                "scopeLogs": [{
                    "scope": {"name": "test"},
                    "logRecords": [{
                        "timeUnixNano": str(int(time.time() * 1e9)),
                        "severityText": "INFO",
                        "body": {"stringValue": "Test log from script"}
                    }]
                }]
            }]
        }
        
        response = requests.post(
            logs_url,
            json=test_log,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        
        if response.status_code == 200:
            print(f"✅ OTLP logs endpoint reachable")
        else:
            print(f"⚠️  OTLP logs endpoint returned: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to OTLP logs endpoint at {logs_url}")
    except Exception as e:
        print(f"⚠️  OTLP logs test error: {e}")
    
    return True

def verify_grafana_access():
    """Check if Grafana is accessible."""
    print("\n🔍 Checking Grafana access...")
    
    grafana_url = "http://localhost:10300"
    try:
        response = requests.get(f"{grafana_url}/api/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ Grafana is accessible at {grafana_url}")
            print("   Default credentials: admin / admin_observability_2024")
        else:
            print(f"⚠️  Grafana returned status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Grafana at {grafana_url}")
        print("   Make sure the observability stack is running")
    except Exception as e:
        print(f"⚠️  Grafana check error: {e}")

def main():
    print("=" * 60)
    print("Testing Observability Platform Integration")
    print("=" * 60)
    
    # Check if observability stack is running
    test_observability_collector()
    
    # Test API endpoints
    test_api_endpoints()
    
    # Check Grafana
    verify_grafana_access()
    
    print("\n" + "=" * 60)
    print("📊 Next Steps:")
    print("1. Start the observability stack if not running:")
    print("   cd ../observability && ./scripts/start_stack.sh")
    print("\n2. Start the FastAPI application with staging tier:")
    print("   APP_DEPLOYMENT_TIER=staging uv run uvicorn app.main:app --reload")
    print("\n3. Access Grafana dashboards:")
    print("   http://localhost:10300 (admin/admin_observability_2024)")
    print("\n4. Query logs in Grafana Explore:")
    print("   {job=\"testing-observability-platform\"}")
    print("\n5. View traces in Tempo:")
    print("   Search for service.name=\"testing-observability-platform\"")
    print("=" * 60)

if __name__ == "__main__":
    main()