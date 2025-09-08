"""
Unit tests for CSV parser utility functions.
"""

import pytest
import io
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from fastapi import UploadFile

from app.utils.csv_parser import CSVParser
from app.schemas.dataset import DatasetItemBase, ConversationTurn


class TestCSVParser:
    """Test cases for the CSVParser class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CSVParser()

    @pytest.mark.asyncio
    async def test_simple_csv_parsing(self):
        """Test parsing simple format CSV."""
        csv_content = """input,expected_output,context,tags
"What is 2+2?","4","math basics","arithmetic,basic"
"What is Python?","A programming language","programming","python,language"
"""
        
        # Create mock UploadFile
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="test.csv", file=file_like)
        
        items, warnings, metadata = await self.parser.parse_csv_file(upload_file)
        
        assert len(items) == 2
        assert len(warnings) == 0
        assert metadata['format_type'] == 'structured'
        assert metadata['total_rows'] == 2
        
        # Check first item
        first_item = items[0]
        assert first_item.input == "What is 2+2?"
        assert first_item.expected_output == "4"
        assert first_item.context == ["math basics"]
        assert first_item.tags == ["arithmetic", "basic"]

    @pytest.mark.asyncio
    async def test_conversation_csv_parsing(self):
        """Test parsing conversation format CSV."""
        csv_content = """role,content,scenario,expected_outcome,test_id
user,"Hello there!",greeting,friendly_response,test_1
assistant,"Hi! How can I help you?",greeting,,test_1
user,"What's the weather?",weather,helpful_info,test_2
assistant,"I don't have access to weather data.",weather,,test_2
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="conversation.csv", file=file_like)
        
        items, warnings, metadata = await self.parser.parse_csv_file(upload_file)
        
        assert len(items) == 2  # Two conversations
        assert metadata['format_type'] == 'conversation'
        
        # Check first conversation
        first_conv = items[0]
        assert isinstance(first_conv.input, dict)
        assert first_conv.input['scenario'] == 'greeting'
        assert len(first_conv.input['turns']) == 2
        
        # Check conversation turns
        first_turn = first_conv.input['turns'][0]
        assert first_turn['role'] == 'user'
        assert first_turn['content'] == 'Hello there!'

    @pytest.mark.asyncio
    async def test_json_conversation_parsing(self):
        """Test parsing JSON conversation format."""
        csv_content = """input,expected_outcome,context
"{""scenario"": ""test conversation"", ""turns"": [{""role"": ""user"", ""content"": ""Hello""}, {""role"": ""assistant"", ""content"": ""Hi there!""}]}","polite_response","test context"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="json_conv.csv", file=file_like)
        
        items, warnings, metadata = await self.parser.parse_csv_file(upload_file)
        
        assert len(items) == 1
        assert metadata['format_type'] == 'structured'
        
        item = items[0]
        # The JSON conversation should be stored as input string
        assert '"scenario"' in str(item.input)
        assert '"turns"' in str(item.input)

    @pytest.mark.asyncio
    async def test_format_detection(self):
        """Test CSV format detection logic."""
        # Test conversation format detection
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp.write("role,content,scenario\nuser,hello,test\n")
            tmp.flush()
            
            format_info = await self.parser._detect_format(tmp.name)
            assert format_info['type'] == 'conversation'
            
            Path(tmp.name).unlink()
        
        # Test structured format detection
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp.write("input,expected_output\nhello,hi\n")
            tmp.flush()
            
            format_info = await self.parser._detect_format(tmp.name)
            assert format_info['type'] == 'structured'
            
            Path(tmp.name).unlink()

    @pytest.mark.asyncio
    async def test_file_size_validation(self):
        """Test file size limit validation."""
        # Create a large content string
        large_content = "input,output\n" + "test,response\n" * 100000
        
        file_like = io.BytesIO(large_content.encode('utf-8'))
        upload_file = UploadFile(filename="large.csv", file=file_like)
        
        # Mock the MAX_FILE_SIZE to be smaller for testing
        original_max_size = self.parser.MAX_FILE_SIZE
        self.parser.MAX_FILE_SIZE = 1000  # 1KB limit
        
        try:
            with pytest.raises(Exception):  # Should raise HTTPException
                await self.parser.parse_csv_file(upload_file)
        finally:
            self.parser.MAX_FILE_SIZE = original_max_size

    def test_parse_single_row(self):
        """Test parsing a single CSV row."""
        row = {
            'input': 'Test question',
            'expected_output': 'Test answer',
            'context': '["background info"]',
            'tags': 'tag1,tag2',
            'meta_domain': 'test'
        }
        
        item = self.parser._parse_single_row(row, 1)
        
        assert item.input == 'Test question'
        assert item.expected_output == 'Test answer'
        assert item.context == ['background info']
        assert item.tags == ['tag1', 'tag2']
        assert item.metadata == {'meta_domain': 'test'}

    def test_parse_single_row_with_alternative_fields(self):
        """Test parsing row with alternative input field names."""
        row = {
            'prompt': 'Test prompt',
            'expected_output': 'Response',
            'question': 'Ignored because prompt takes precedence'
        }
        
        item = self.parser._parse_single_row(row, 1)
        assert item.input == 'Test prompt'

    def test_parse_single_row_missing_input(self):
        """Test handling row with missing input field."""
        row = {
            'expected_output': 'Response',
            'context': 'some context'
        }
        
        item = self.parser._parse_single_row(row, 1)
        assert item is None

    def test_convert_conversations_to_items(self):
        """Test converting conversation dictionaries to DatasetItems."""
        conversations = {
            'conv1': {
                'scenario': 'Test conversation',
                'turns': [
                    ConversationTurn(role='user', content='Hello'),
                    ConversationTurn(role='assistant', content='Hi there!')
                ],
                'expected_outcome': 'Friendly response',
                'metadata': {'test_key': 'test_value'},
                'tags': {'tag1', 'tag2'}
            }
        }
        
        items = self.parser._convert_conversations_to_items(conversations)
        
        assert len(items) == 1
        item = items[0]
        assert item.input['scenario'] == 'Test conversation'
        assert len(item.input['turns']) == 2
        assert item.expected_outcome == 'Friendly response'
        assert item.tags == ['tag1', 'tag2']

    def test_convert_conversations_empty_turns(self):
        """Test handling conversations with no turns."""
        conversations = {
            'empty_conv': {
                'scenario': 'Empty',
                'turns': [],
                'expected_outcome': '',
                'metadata': {},
                'tags': set()
            }
        }
        
        items = self.parser._convert_conversations_to_items(conversations)
        assert len(items) == 0
        assert len(self.parser.warnings) > 0

    @pytest.mark.asyncio
    async def test_validation_only_mode(self):
        """Test CSV validation without full parsing."""
        csv_content = """input,expected_output
"Test input","Test output"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="validate.csv", file=file_like)
        
        items, warnings, metadata = await self.parser.parse_csv_file(upload_file, validate_only=True)
        
        # In validation mode, items might be processed differently
        assert isinstance(items, list)
        assert isinstance(warnings, list)
        assert isinstance(metadata, dict)

    @pytest.mark.asyncio
    async def test_malformed_csv(self):
        """Test handling of malformed CSV content."""
        malformed_content = """input,output
"unclosed quote, test
"normal","value"
"""
        
        file_like = io.BytesIO(malformed_content.encode('utf-8'))
        upload_file = UploadFile(filename="malformed.csv", file=file_like)
        
        # Should handle gracefully and provide warnings
        items, warnings, metadata = await self.parser.parse_csv_file(upload_file)
        
        # Depending on pandas behavior, might succeed with warnings or fail gracefully
        assert isinstance(items, list)
        assert isinstance(warnings, list)

    def test_batch_validate(self):
        """Test batch validation of items."""
        items = [
            DatasetItemBase(input="valid input", expected_output="output"),
            DatasetItemBase(input="another valid", tags=["test"])
        ]
        
        validated = self.parser._batch_validate(items)
        assert len(validated) == 2

    @pytest.mark.asyncio 
    async def test_empty_csv_file(self):
        """Test handling of empty CSV file."""
        empty_content = ""
        
        file_like = io.BytesIO(empty_content.encode('utf-8'))
        upload_file = UploadFile(filename="empty.csv", file=file_like)
        
        items, warnings, metadata = await self.parser.parse_csv_file(upload_file)
        
        assert len(items) == 0
        assert len(warnings) > 0  # Should have warning about empty file

    @pytest.mark.asyncio
    async def test_csv_with_only_headers(self):
        """Test CSV file with headers but no data rows."""
        headers_only = "input,expected_output,context\n"
        
        file_like = io.BytesIO(headers_only.encode('utf-8'))
        upload_file = UploadFile(filename="headers_only.csv", file=file_like)
        
        items, warnings, metadata = await self.parser.parse_csv_file(upload_file)
        
        assert len(items) == 0
        assert metadata['total_rows'] == 0


@pytest.fixture
def sample_simple_csv():
    """Fixture for sample simple format CSV content."""
    return """input,expected_output,context,tags,meta_domain
"What is AI?","Artificial Intelligence","technology","ai,tech","technology"
"Define ML","Machine Learning","computer science","ml,ai","technology"
"""


@pytest.fixture
def sample_conversation_csv():
    """Fixture for sample conversation format CSV content."""
    return """role,content,scenario,expected_outcome,test_id
user,"Hello!",greeting,friendly_greeting,conv1
assistant,"Hi there! How can I help?",greeting,,conv1
user,"What's your name?",introduction,name_response,conv2
assistant,"I'm Claude, an AI assistant.",introduction,,conv2
"""


@pytest.fixture
def sample_json_conversation_csv():
    """Fixture for JSON conversation format CSV."""
    return """input,expected_outcome,context,meta_type
"{""scenario"": ""Customer service"", ""turns"": [{""role"": ""user"", ""content"": ""I have a problem""}, {""role"": ""assistant"", ""content"": ""How can I help you?""}]}","helpful_response","customer support","conversational"
"""


class TestCSVParserIntegration:
    """Integration tests for CSV parser with real file scenarios."""

    @pytest.mark.asyncio
    async def test_real_healthcare_conversation_format(self, sample_conversation_csv):
        """Test with realistic healthcare conversation data."""
        parser = CSVParser()
        
        file_like = io.BytesIO(sample_conversation_csv.encode('utf-8'))
        upload_file = UploadFile(filename="healthcare.csv", file=file_like)
        
        items, warnings, metadata = await parser.parse_csv_file(upload_file)
        
        assert len(items) == 2  # Two conversations
        assert metadata['format_type'] == 'conversation'
        
        # Verify conversation structure
        for item in items:
            assert 'scenario' in item.input
            assert 'turns' in item.input
            assert len(item.input['turns']) == 2

    @pytest.mark.asyncio
    async def test_mixed_metadata_extraction(self, sample_simple_csv):
        """Test extraction of metadata fields."""
        parser = CSVParser()
        
        file_like = io.BytesIO(sample_simple_csv.encode('utf-8'))
        upload_file = UploadFile(filename="metadata_test.csv", file=file_like)
        
        items, warnings, metadata = await parser.parse_csv_file(upload_file)
        
        assert len(items) == 2
        
        # Check metadata extraction
        for item in items:
            assert 'meta_domain' in item.metadata
            assert item.metadata['meta_domain'] == 'technology'

    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self):
        """Test performance with a moderately large CSV."""
        parser = CSVParser()
        
        # Create a CSV with many rows
        header = "input,expected_output,context\n"
        rows = "\n".join([f'"Question {i}","Answer {i}","Context {i}"' for i in range(1000)])
        large_csv = header + rows
        
        file_like = io.BytesIO(large_csv.encode('utf-8'))
        upload_file = UploadFile(filename="large_test.csv", file=file_like)
        
        items, warnings, metadata = await parser.parse_csv_file(upload_file)
        
        assert len(items) == 1000
        assert metadata['total_rows'] == 1000
        assert len(warnings) == 0  # Should process without warnings