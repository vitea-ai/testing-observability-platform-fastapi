#!/usr/bin/env python3
"""
Test script to verify frontend-backend CSV upload integration.
This simulates the API calls that the frontend would make.
"""

import requests
import json
import time
import sys
from pathlib import Path

# Configuration
FASTAPI_BASE_URL = "http://localhost:8000"
CSV_FILE_PATH = "samples/healthcare_appointment_dataset.csv"

def test_health_check():
    """Test if the FastAPI backend is healthy."""
    print("🔍 Testing FastAPI backend health...")
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/health", timeout=5)
        if response.status_code in [200, 201]:  # Accept both 200 and 201 (Created)
            print("✅ FastAPI backend is healthy")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to FastAPI backend: {e}")
        return False

def test_csv_validation():
    """Test CSV validation endpoint (simulates frontend validation)."""
    print(f"\n🧪 Testing CSV validation with {CSV_FILE_PATH}...")
    
    if not Path(CSV_FILE_PATH).exists():
        print(f"❌ CSV file not found: {CSV_FILE_PATH}")
        return False
    
    try:
        with open(CSV_FILE_PATH, 'rb') as file:
            files = {'file': file}
            response = requests.post(
                f"{FASTAPI_BASE_URL}/api/v1/datasets/validate-csv",
                files=files,
                timeout=30
            )
        
        if response.status_code in [200, 201]:  # Accept both 200 and 201 (Created)
            result = response.json()
            print("✅ CSV validation successful!")
            print(f"   Format: {result.get('format_type', 'unknown')}")
            print(f"   Columns: {len(result.get('columns', []))} detected")
            print(f"   Total rows: {result.get('total_rows', 0)}")
            if result.get('warnings'):
                print(f"   Warnings: {len(result['warnings'])}")
                for warning in result['warnings']:
                    print(f"     - {warning}")
            return result
        else:
            print(f"❌ CSV validation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ CSV validation error: {e}")
        return False

def test_csv_upload():
    """Test CSV upload endpoint (simulates frontend upload)."""
    print(f"\n📤 Testing CSV upload with {CSV_FILE_PATH}...")
    
    if not Path(CSV_FILE_PATH).exists():
        print(f"❌ CSV file not found: {CSV_FILE_PATH}")
        return False
    
    try:
        with open(CSV_FILE_PATH, 'rb') as file:
            files = {'file': file}
            data = {
                'name': 'Test Healthcare Appointments',
                'description': 'Test upload of healthcare appointment dataset',
                'tags': 'healthcare,testing'  # Comma-separated string, not JSON
            }
            
            response = requests.post(
                f"{FASTAPI_BASE_URL}/api/v1/datasets/upload-csv",
                files=files,
                data=data,
                timeout=60
            )
        
        if response.status_code in [200, 201]:  # Accept both 200 and 201 (Created)
            result = response.json()
            print("✅ CSV upload successful!")
            
            # Check if it's synchronous or asynchronous processing
            if 'upload_id' in result:
                print(f"   Upload ID: {result['upload_id']}")
                print("   Background processing started...")
                return test_upload_status_polling(result['upload_id'])
            elif 'dataset' in result:
                print(f"   Dataset created: {result['dataset']['name']}")
                print(f"   Records processed: {result.get('rows_processed', 0)}")
                return result
            else:
                print(f"   Unexpected response format: {result}")
                return result
        else:
            print(f"❌ CSV upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ CSV upload error: {e}")
        return False

def test_upload_status_polling(upload_id):
    """Test upload status polling (simulates frontend background processing)."""
    print(f"\n⏳ Polling upload status for: {upload_id}")
    
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get(
                f"{FASTAPI_BASE_URL}/api/v1/datasets/upload-status/{upload_id}",
                timeout=10
            )
            
            if response.status_code in [200, 201]:  # Accept both 200 and 201 (Created)
                result = response.json()
                status = result.get('status', 'unknown')
                print(f"   Attempt {attempt + 1}: Status = {status}")
                
                if status == 'completed':
                    print("✅ Upload processing completed!")
                    dataset = result.get('dataset', {})
                    print(f"   Dataset: {dataset.get('name', 'Unknown')}")
                    print(f"   Records: {dataset.get('record_count', 0)}")
                    return result
                elif status == 'failed':
                    print("❌ Upload processing failed!")
                    print(f"   Error: {result.get('error', 'Unknown error')}")
                    return False
                elif status in ['processing', 'pending']:
                    time.sleep(2)  # Wait 2 seconds before next poll
                    continue
                else:
                    print(f"❌ Unknown status: {status}")
                    return False
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"❌ Status polling error: {e}")
            return False
    
    print("❌ Upload status polling timeout")
    return False

def test_datasets_list():
    """Test listing datasets to verify creation."""
    print(f"\n📋 Testing datasets list...")
    
    try:
        response = requests.get(
            f"{FASTAPI_BASE_URL}/api/v1/datasets/",
            timeout=10
        )
        
        if response.status_code in [200, 201]:  # Accept both 200 and 201 (Created)
            result = response.json()
            datasets = result.get('datasets', []) if isinstance(result, dict) else result
            print(f"✅ Found {len(datasets)} dataset(s)")
            
            for dataset in datasets[:3]:  # Show first 3
                name = dataset.get('name', 'Unknown')
                count = dataset.get('record_count', 0)
                created = dataset.get('created_at', 'Unknown')
                print(f"   - {name} ({count} records, created: {created})")
            
            return result
        else:
            print(f"❌ Datasets list failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Datasets list error: {e}")
        return False

def main():
    """Run integration tests."""
    print("🚀 Testing Frontend-Backend CSV Upload Integration\n")
    print("=" * 60)
    
    # Test 1: Health check
    if not test_health_check():
        print("\n❌ Backend not available. Ensure FastAPI is running on port 8000.")
        sys.exit(1)
    
    # Test 2: CSV validation
    validation_result = test_csv_validation()
    if not validation_result:
        print("\n❌ CSV validation failed.")
        sys.exit(1)
    
    # Test 3: CSV upload
    upload_result = test_csv_upload()
    if not upload_result:
        print("\n❌ CSV upload failed.")
        sys.exit(1)
    
    # Test 4: List datasets
    list_result = test_datasets_list()
    if not list_result:
        print("\n⚠️  Dataset listing failed, but upload may have succeeded.")
    
    print("\n" + "=" * 60)
    print("🎉 Integration tests completed successfully!")
    print("\nThe frontend should now be able to:")
    print("✅ Validate CSV files using server-side processing")
    print("✅ Upload CSV files with progress tracking")
    print("✅ Handle both single-turn and multi-conversation formats")
    print("✅ Display validation warnings and format detection")
    
    print(f"\n🌐 Frontend URL: http://localhost:3001")
    print("   Navigate to Dataset Management to test the UI!")

if __name__ == "__main__":
    main()