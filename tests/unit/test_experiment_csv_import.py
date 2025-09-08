"""
Unit tests for experiment CSV import functionality.
"""

import pytest
import io
from unittest.mock import Mock, AsyncMock
from fastapi import UploadFile

from app.utils.csv_parser import CSVParser


class TestExperimentCSVImport:
    """Test cases for experiment-specific CSV parsing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CSVParser()

    @pytest.mark.asyncio
    async def test_parse_experiment_results_basic(self):
        """Test parsing basic experiment results CSV."""
        csv_content = """test_id,input,expected_output,actual_output,latency_ms,status
test_1,"What is 2+2?","4","4",125,completed
test_2,"Capital of France?","Paris","Paris",89,completed
"""
        
        # Create mock UploadFile
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="results.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        assert len(results) == 2
        assert len(warnings) == 0
        assert metadata['format_type'] == 'experiment_results'
        assert metadata['total_rows'] == 2
        
        # Check first result
        first_result = results[0]
        assert first_result['test_id'] == 'test_1'
        assert first_result['input'] == "What is 2+2?"
        assert first_result['expected_output'] == "4"
        assert first_result['actual_output'] == "4"
        assert first_result['execution_time'] == 0.125
        assert first_result['status'] == 'completed'

    @pytest.mark.asyncio
    async def test_parse_experiment_with_token_usage(self):
        """Test parsing experiment results with token usage metrics."""
        csv_content = """test_id,input,actual_output,token_usage_input,token_usage_output,execution_time
test_1,"Hello","Hi there!",5,10,0.5
test_2,"Goodbye","See you later!",6,12,0.3
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="results.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        assert len(results) == 2
        
        # Check token usage
        first_result = results[0]
        assert first_result['token_usage'] is not None
        assert first_result['token_usage']['input'] == 5
        assert first_result['token_usage']['output'] == 10
        assert first_result['execution_time'] == 0.5

    @pytest.mark.asyncio
    async def test_parse_experiment_with_metadata(self):
        """Test parsing experiment results with metadata fields."""
        csv_content = """test_id,input,actual_output,meta_model,meta_temperature,meta_provider
test_1,"Test input","Test output","gpt-4","0.7","openai"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="results.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        assert len(results) == 1
        
        # Check metadata extraction
        first_result = results[0]
        assert first_result['meta_data']['meta_model'] == "gpt-4"
        assert str(first_result['meta_data']['meta_temperature']) == "0.7"
        assert first_result['meta_data']['meta_provider'] == "openai"

    @pytest.mark.asyncio
    async def test_parse_experiment_with_json_context(self):
        """Test parsing experiment results with JSON context field."""
        csv_content = """test_id,input,actual_output,context,retrieval_context
test_1,"Question","Answer","[""doc1"", ""doc2""]","[{""id"": 1, ""text"": ""relevant doc""}]"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="results.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        assert len(results) == 1
        
        # Check JSON parsing
        first_result = results[0]
        assert first_result['context'] == ["doc1", "doc2"]
        assert first_result['retrieval_context'] == [{"id": 1, "text": "relevant doc"}]

    @pytest.mark.asyncio
    async def test_parse_experiment_alternative_columns(self):
        """Test parsing with alternative column names."""
        csv_content = """test_case_id,prompt,output,latency_ms
tc_1,"What's the weather?","It's sunny today",200
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="results.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        assert len(results) == 1
        
        # Check alternative column mapping
        first_result = results[0]
        assert first_result['test_id'] == 'tc_1'
        assert first_result['input'] == "What's the weather?"
        assert first_result['actual_output'] == "It's sunny today"
        assert first_result['execution_time'] == 0.2

    @pytest.mark.asyncio
    async def test_parse_experiment_missing_input_field(self):
        """Test handling of missing input field."""
        csv_content = """test_id,expected_output,actual_output
test_1,"4","4"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="results.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        # Should have warning about missing input
        assert len(results) == 0
        assert len(warnings) > 0
        assert "No input field found" in warnings[0]

    @pytest.mark.asyncio
    async def test_parse_experiment_validate_only(self):
        """Test validation-only mode."""
        csv_content = """test_id,input,actual_output
""" + "\n".join([f'test_{i},"Input {i}","Output {i}"' for i in range(200)])
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="results.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(
            upload_file, 
            validate_only=True
        )
        
        # Should only parse sample size in validation mode
        assert len(results) <= self.parser.SAMPLE_SIZE
        assert metadata['total_rows'] == 200

    @pytest.mark.asyncio
    async def test_parse_experiment_with_errors(self):
        """Test parsing with error field."""
        csv_content = """test_id,input,actual_output,status,error
test_1,"Bad input","","failed","API timeout"
test_2,"Good input","Output","completed",""
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="results.csv", file=file_like)
        
        results, warnings, metadata = await self.parser.parse_experiment_results(upload_file)
        
        assert len(results) == 2
        
        # Check error handling
        assert results[0]['status'] == 'failed'
        assert results[0]['error'] == "API timeout"
        assert results[1]['status'] == 'completed'
        # Empty string in CSV becomes NaN in pandas, which we handle as None
        assert results[1]['error'] in ["", None] or (isinstance(results[1]['error'], float) and str(results[1]['error']) == 'nan')