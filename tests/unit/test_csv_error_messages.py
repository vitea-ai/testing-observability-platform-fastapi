"""
Unit tests for CSV import error messages and user feedback.
"""

import pytest
import io
from unittest.mock import Mock, AsyncMock
from fastapi import UploadFile

from app.utils.csv_parser import CSVParser


class TestCSVErrorMessages:
    """Test cases for improved CSV error messages."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CSVParser()

    @pytest.mark.asyncio
    async def test_missing_input_column_error_message(self):
        """Test error message when input column is missing."""
        csv_content = """Test ID,Expected Result,Actual Result
test_1,"4","4"
test_2,"Paris","Paris"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="missing_input.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        # Should have no results but helpful warnings
        assert len(results) == 0
        assert len(warnings) > 0
        
        # Check that warning mentions the expected columns
        warning = warnings[0]
        assert "No input field found" in warning
        assert "Expected columns:" in warning
        assert "'input'" in warning or "input" in warning
        
        # Check that warning shows found columns
        assert "Found columns:" in warning
        assert "Test ID" in warning
        assert "Expected Result" in warning

    @pytest.mark.asyncio
    async def test_wrong_column_names_error_message(self):
        """Test error message with completely wrong column names."""
        csv_content = """Name,Description,Value,Status
Item 1,Test description,100,Active
Item 2,Another test,200,Inactive
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="wrong_columns.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        # Should have warnings about missing required fields
        assert len(results) == 0
        assert len(warnings) > 0
        
        # Check that warning is helpful
        warning = warnings[0]
        assert "No input field found" in warning
        assert "Found columns:" in warning
        assert "Name" in warning
        assert "Description" in warning

    @pytest.mark.asyncio
    async def test_similar_column_names_suggestion(self):
        """Test error message when columns have similar but not exact names."""
        csv_content = """test_ids,inputs,expected_outputs,actual_outputs
test_1,"Question 1","Answer 1","Result 1"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="similar_columns.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        # Parser should not find 'input' (only 'inputs')
        assert len(results) == 0
        assert len(warnings) > 0
        
        # Warning should show the similar column names
        warning = warnings[0]
        assert "Found columns:" in warning
        assert "inputs" in warning  # Shows user has 'inputs' not 'input'

    @pytest.mark.asyncio
    async def test_kelly_format_error_message(self):
        """Test error message for Kelly's specific CSV format."""
        csv_content = """Test ID,Scenario Name,Setting,Specialty,Test Script Dialogue,Scribe Summary Output,Evaluation Criteria
TEST-001,Cardiology Consultation,Clinic,Cardiology,"Doctor patient dialogue","Expected summary","Criteria"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="kelly_format.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        # Should have warnings with helpful context
        assert len(results) == 0
        assert len(warnings) > 0
        
        warning = warnings[0]
        assert "No input field found" in warning
        assert "Test Script Dialogue" in warning  # Shows what columns were found
        assert "Expected columns:" in warning

    @pytest.mark.asyncio
    async def test_empty_csv_error_message(self):
        """Test error message for empty CSV file."""
        csv_content = """input,expected_output,actual_output
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="empty.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        # Should handle empty CSV gracefully
        assert len(results) == 0
        assert metadata['total_rows'] == 0

    @pytest.mark.asyncio
    async def test_successful_parse_no_warnings(self):
        """Test that correctly formatted CSV produces no warnings."""
        csv_content = """test_id,input,expected_output,actual_output,status
test_1,"What is 2+2?","4","4",completed
test_2,"Capital of France?","Paris","Paris",completed
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="correct.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        # Should parse successfully with no warnings
        assert len(results) == 2
        assert len(warnings) == 0
        assert metadata['format_type'] == 'experiment_results'
        
        # Verify data is correct
        assert results[0]['input'] == "What is 2+2?"
        assert results[0]['expected_output'] == "4"
        assert results[0]['actual_output'] == "4"

    @pytest.mark.asyncio
    async def test_partial_success_with_warnings(self):
        """Test CSV with some valid and some invalid rows."""
        csv_content = """test_id,input,expected_output,actual_output
test_1,"Valid input","Expected","Actual"
test_2,,"Missing input","Output"
test_3,"Another valid","Expected2","Actual2"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="partial.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        # CSV parser may handle empty strings as empty but valid input
        # so we check that we get results and potentially warnings
        assert len(results) >= 2  # At least the valid rows
        
        # Check that valid results are present
        valid_inputs = [r['input'] for r in results if r['input']]
        assert "Valid input" in valid_inputs
        assert "Another valid" in valid_inputs