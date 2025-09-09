"""
Unit tests for intelligent column name mapping in CSV parser.
"""

import pytest
import io
from unittest.mock import Mock, AsyncMock
from fastapi import UploadFile

from app.utils.csv_parser import CSVParser


class TestIntelligentColumnMapping:
    """Test cases for intelligent column name recognition."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CSVParser()

    @pytest.mark.asyncio
    async def test_expected_outcome_maps_to_expected_output(self):
        """Test that 'expected_outcome' column is recognized as expected output."""
        csv_content = """test_id,input,expected_outcome,actual_output
test_1,"What is 2+2?","The answer is 4","4"
test_2,"Capital of France?","The capital is Paris","Paris"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="expected_outcome.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        assert len(results) == 2
        assert len(warnings) == 0
        
        # Check that expected_outcome was mapped to expected_output
        assert results[0]['expected_output'] == "The answer is 4"
        assert results[1]['expected_output'] == "The capital is Paris"

    @pytest.mark.asyncio
    async def test_actual_result_maps_to_actual_output(self):
        """Test that columns containing 'actual' are recognized."""
        csv_content = """test_id,input,expected_output,actual_result
test_1,"Question 1","Expected 1","Result 1"
test_2,"Question 2","Expected 2","Result 2"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="actual_result.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        assert len(results) == 2
        
        # Check that actual_result was mapped to actual_output
        assert results[0]['actual_output'] == "Result 1"
        assert results[1]['actual_output'] == "Result 2"

    @pytest.mark.asyncio
    async def test_expected_response_maps_correctly(self):
        """Test that any column with 'expected' in the name is recognized."""
        csv_content = """test_id,input,expected_response,actual_response
test_1,"Input 1","Expected response 1","Actual response 1"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="response_columns.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        assert len(results) == 1
        
        # Check that expected_response was mapped to expected_output
        assert results[0]['expected_output'] == "Expected response 1"
        assert results[0]['actual_output'] == "Actual response 1"

    @pytest.mark.asyncio
    async def test_kelly_cardiology_format(self):
        """Test Kelly's specific cardiology CSV format."""
        csv_content = """input,expected_outcome,actual_output,meta_weight,context
"Patient consultation","Referred by PCP for evaluation","Full medical documentation","High","Medical context"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="kelly_format.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        assert len(results) == 1
        assert len(warnings) == 0
        
        # Check that Kelly's format is properly handled
        result = results[0]
        assert result['input'] == "Patient consultation"
        assert result['expected_output'] == "Referred by PCP for evaluation"
        assert result['actual_output'] == "Full medical documentation"
        assert result['meta_data']['meta_weight'] == "High"

    @pytest.mark.asyncio
    async def test_output_column_fallback(self):
        """Test that 'output' column is used as fallback when no 'actual' column exists."""
        csv_content = """test_id,input,expected_output,output
test_1,"Question","Expected","Output value"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="output_fallback.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        assert len(results) == 1
        
        # Check that output was used as fallback for actual_output
        assert results[0]['actual_output'] == "Output value"

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self):
        """Test that column matching is case-insensitive."""
        csv_content = """test_id,input,Expected_Output,ACTUAL_OUTPUT
test_1,"Input","Expected","Actual"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="case_insensitive.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        assert len(results) == 1
        
        # Check case-insensitive matching
        assert results[0]['expected_output'] == "Expected"
        assert results[0]['actual_output'] == "Actual"

    @pytest.mark.asyncio
    async def test_no_expected_column_returns_empty(self):
        """Test that missing expected column returns empty string."""
        csv_content = """test_id,input,actual_output
test_1,"Input","Actual"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="no_expected.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        assert len(results) == 1
        
        # Check that missing expected_output is empty string
        assert results[0]['expected_output'] == ""
        assert results[0]['actual_output'] == "Actual"

    @pytest.mark.asyncio
    async def test_multiple_expected_columns_uses_first(self):
        """Test that when multiple 'expected' columns exist, the first is used."""
        csv_content = """test_id,input,expected_output,expected_result,actual_output
test_1,"Input","First expected","Second expected","Actual"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="multiple_expected.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        assert len(results) == 1
        
        # Should use the first column containing 'expected'
        assert results[0]['expected_output'] == "First expected"