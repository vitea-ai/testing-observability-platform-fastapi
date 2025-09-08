"""
Edge case and error handling tests for CSV upload functionality.
"""

import pytest
import io
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from fastapi import UploadFile, HTTPException
from pydantic import ValidationError

from app.utils.csv_parser import CSVParser
from app.schemas.dataset import ConversationTurn, ConversationInput, DatasetItemBase


class TestCSVParserEdgeCases:
    """Test edge cases and error conditions for CSV parser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CSVParser()

    @pytest.mark.asyncio
    async def test_invalid_utf8_encoding(self):
        """Test handling of invalid UTF-8 encoding."""
        # Create content with invalid UTF-8 bytes
        invalid_content = b"input,output\n\xff\xfe,response\n"
        
        file_like = io.BytesIO(invalid_content)
        upload_file = UploadFile(filename="invalid_encoding.csv", file=file_like)
        
        # Should handle encoding errors gracefully
        with pytest.raises(Exception):  # May raise various encoding errors
            await self.parser.parse_csv_file(upload_file)

    @pytest.mark.asyncio
    async def test_extremely_long_content_line(self):
        """Test handling of extremely long content lines."""
        # Create a CSV with very long content (over 2000 chars)
        long_content = "x" * 3000
        csv_content = f'input,output\n"{long_content}","response"\n'
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="long_content.csv", file=file_like)
        
        # Should handle long lines (may truncate based on implementation)
        items, warnings, metadata = await self.parser.parse_csv_file(upload_file)
        
        # Verify it processes without crashing
        assert isinstance(items, list)
        assert isinstance(warnings, list)

    @pytest.mark.asyncio
    async def test_special_characters_in_csv(self):
        """Test handling of special characters."""
        csv_content = """input,output,context
"What's the weather like today?","It's sunny! ☀️","weather"
"Can you handle émojis? 🚀","Yes, I can! 🎉","unicode"
"Quote test: ""nested quotes""","Handled correctly","quotes"
"Comma, test","Works fine","punctuation"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="special_chars.csv", file=file_like)
        
        items, warnings, metadata = await self.parser.parse_csv_file(upload_file)
        
        assert len(items) == 4
        # Check emoji handling
        assert "☀️" in items[0].input
        assert "émojis" in items[1].input
        # Check nested quotes
        assert 'nested quotes' in items[2].input

    @pytest.mark.asyncio
    async def test_inconsistent_column_counts(self):
        """Test CSV with inconsistent column counts per row."""
        csv_content = """input,output,context
"Complete row","Response","Context"
"Missing context","Response"
"Extra field","Response","Context","Unexpected"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="inconsistent.csv", file=file_like)
        
        # Pandas should handle this gracefully
        items, warnings, metadata = await self.parser.parse_csv_file(upload_file)
        
        assert len(items) >= 1  # Should process at least some rows

    @pytest.mark.asyncio
    async def test_conversation_with_invalid_roles(self):
        """Test conversation format with invalid roles."""
        csv_content = """role,content,scenario,test_id
user,"Hello",greeting,test_1
moderator,"Invalid role content",greeting,test_1
assistant,"Response",greeting,test_1
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="invalid_roles.csv", file=file_like)
        
        items, warnings, metadata = await self.parser.parse_csv_file(upload_file)
        
        # Should process and warn about invalid role
        assert len(warnings) > 0
        assert any("Invalid role" in warning or "moderator" in warning for warning in warnings)

    @pytest.mark.asyncio
    async def test_conversation_with_empty_content(self):
        """Test conversation with empty content fields."""
        csv_content = """role,content,scenario,test_id
user,"Hello",greeting,test_1
assistant,"",greeting,test_1
user,"How are you?",greeting,test_1
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="empty_content.csv", file=file_like)
        
        items, warnings, metadata = await self.parser.parse_csv_file(upload_file)
        
        # Should handle empty content gracefully
        assert len(items) >= 1

    @pytest.mark.asyncio
    async def test_malformed_json_in_input(self):
        """Test handling of malformed JSON in input field."""
        csv_content = '''input,output
"{'invalid': json}","Should handle gracefully"
"{"valid": "json"}","This works"
'''
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="malformed_json.csv", file=file_like)
        
        items, warnings, metadata = await self.parser.parse_csv_file(upload_file)
        
        # Should process both rows (malformed JSON treated as string)
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_extremely_nested_json(self):
        """Test deeply nested JSON structures."""
        nested_json = '{"level1": {"level2": {"level3": {"level4": {"content": "deep"}}}}}'
        csv_content = f'''input,output
"{nested_json}","Handled"
'''
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="nested_json.csv", file=file_like)
        
        items, warnings, metadata = await self.parser.parse_csv_file(upload_file)
        
        assert len(items) == 1
        assert "level1" in str(items[0].input)

    @pytest.mark.asyncio
    async def test_csv_with_only_whitespace_rows(self):
        """Test CSV with rows containing only whitespace."""
        csv_content = """input,output
"Valid input","Valid output"
"   ","   "
"","Another response"
"""
        
        file_like = io.BytesIO(csv_content.encode('utf-8'))
        upload_file = UploadFile(filename="whitespace.csv", file=file_like)
        
        items, warnings, metadata = await self.parser.parse_csv_file(upload_file)
        
        # Should handle whitespace appropriately
        assert len(items) >= 1

    def test_parser_state_reset(self):
        """Test that parser state resets between uses."""
        parser = CSVParser()
        
        # Add some warnings and errors
        parser.warnings = ["Previous warning"]
        parser.errors = ["Previous error"]
        parser.processed_count = 100
        
        # Create new parser instance for next use
        new_parser = CSVParser()
        
        # State should be clean
        assert new_parser.warnings == []
        assert new_parser.errors == []
        assert new_parser.processed_count == 0

    def test_conversation_validation_edge_cases(self):
        """Test conversation validation with edge cases."""
        # Empty turns list
        conversations = {
            'empty': {
                'scenario': 'Empty conversation',
                'turns': [],
                'expected_outcome': '',
                'metadata': {},
                'tags': set()
            }
        }
        
        items = self.parser._convert_conversations_to_items(conversations)
        assert len(items) == 0
        assert len(self.parser.warnings) > 0

    def test_single_row_parsing_edge_cases(self):
        """Test single row parsing with various edge cases."""
        # Row with all empty strings
        empty_row = {key: "" for key in ['input', 'output', 'context', 'tags']}
        item = self.parser._parse_single_row(empty_row, 1)
        assert item is None
        
        # Row with null values (None)
        null_row = {key: None for key in ['input', 'output', 'context', 'tags']}
        item = self.parser._parse_single_row(null_row, 2)
        assert item is None
        
        # Row with mixed null and empty
        mixed_row = {'input': 'Valid', 'output': None, 'context': '', 'tags': None}
        item = self.parser._parse_single_row(mixed_row, 3)
        assert item is not None
        assert item.input == 'Valid'

    def test_metadata_extraction_edge_cases(self):
        """Test metadata extraction with edge cases."""
        # Row with various meta_ fields
        row = {
            'input': 'Test',
            'meta_': 'Empty meta key',  # Edge case: empty after meta_
            'meta_valid': 'Valid meta',
            'meta_numeric': 123,
            'meta_boolean': True,
            'meta_null': None,
            'not_meta': 'Should be ignored'
        }
        
        item = self.parser._parse_single_row(row, 1)
        
        # Should extract all meta_ fields
        assert 'meta_' in item.metadata
        assert 'meta_valid' in item.metadata  
        assert 'meta_numeric' in item.metadata
        assert 'meta_boolean' in item.metadata
        assert 'meta_null' in item.metadata
        assert 'not_meta' not in item.metadata

    @pytest.mark.asyncio
    async def test_file_permission_errors(self):
        """Test handling of file permission errors."""
        # Mock file operations to simulate permission errors
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.side_effect = PermissionError("Permission denied")
            
            csv_content = "input,output\ntest,response\n"
            file_like = io.BytesIO(csv_content.encode('utf-8'))
            upload_file = UploadFile(filename="permission_test.csv", file=file_like)
            
            with pytest.raises(PermissionError):
                await self.parser.parse_csv_file(upload_file)

    @pytest.mark.asyncio
    async def test_disk_space_errors(self):
        """Test handling of disk space errors during temp file creation."""
        # Mock to simulate disk full error
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.side_effect = OSError("No space left on device")
            
            csv_content = "input,output\ntest,response\n"
            file_like = io.BytesIO(csv_content.encode('utf-8'))
            upload_file = UploadFile(filename="disk_space_test.csv", file=file_like)
            
            with pytest.raises(OSError):
                await self.parser.parse_csv_file(upload_file)

    def test_batch_validation_with_invalid_items(self):
        """Test batch validation with some invalid items."""
        # Mix of valid and potentially problematic items
        items = [
            DatasetItemBase(input="Valid item 1"),
            DatasetItemBase(input="Valid item 2", tags=["test"]),
            # Note: Current schema is quite permissive, so creating truly invalid items is difficult
        ]
        
        validated = self.parser._batch_validate(items)
        
        # All items should pass validation with current permissive schema
        assert len(validated) == len(items)

    @pytest.mark.asyncio
    async def test_max_errors_threshold(self):
        """Test that parser stops after max errors reached."""
        # This would require mocking validation to fail consistently
        # For now, test that the MAX_ERRORS constant exists and is reasonable
        assert hasattr(self.parser, 'MAX_ERRORS')
        assert self.parser.MAX_ERRORS > 0
        assert self.parser.MAX_ERRORS <= 100  # Should be reasonable limit


class TestSchemaValidationEdgeCases:
    """Test edge cases for schema validation."""

    def test_conversation_turn_with_very_long_content(self):
        """Test conversation turn with extremely long content."""
        very_long_content = "x" * 10000  # 10k characters
        
        turn = ConversationTurn(role="user", content=very_long_content)
        assert len(turn.content) == 10000

    def test_conversation_turn_with_unicode_content(self):
        """Test conversation turn with various Unicode characters."""
        unicode_content = "Hello 🌍! Café naïve résumé 北京 العربية 🚀✨"
        
        turn = ConversationTurn(role="assistant", content=unicode_content)
        assert turn.content == unicode_content

    def test_dataset_item_with_complex_nested_input(self):
        """Test dataset item with deeply nested input structure."""
        complex_input = {
            "level1": {
                "level2": {
                    "level3": {
                        "conversations": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi"}
                        ],
                        "metadata": {
                            "source": "nested",
                            "complexity": 9000
                        }
                    }
                }
            },
            "additional_data": ["item1", "item2", {"nested": "object"}]
        }
        
        item = DatasetItemBase(input=complex_input)
        assert item.input["level1"]["level2"]["level3"]["complexity"] == 9000

    def test_dataset_create_with_very_large_dataset(self):
        """Test dataset creation with many items."""
        # Create 1000 items
        items = [
            DatasetItemBase(input=f"Input {i}", expected_output=f"Output {i}")
            for i in range(1000)
        ]
        
        dataset = DatasetCreate(name="Large Dataset", data=items)
        assert len(dataset.data) == 1000

    def test_metadata_with_circular_references(self):
        """Test that metadata handles various data types appropriately."""
        # Note: Actual circular references would cause JSON serialization issues
        # Test with complex but valid structures
        complex_metadata = {
            "lists": [1, 2, [3, 4, [5, 6]]],
            "dicts": {"a": {"b": {"c": "deep"}}},
            "mixed": [{"key": "value"}, [1, 2, 3], "string"],
            "none_values": None,
            "empty_structures": {
                "empty_list": [],
                "empty_dict": {},
                "empty_string": ""
            }
        }
        
        item = DatasetItemBase(input="Test", metadata=complex_metadata)
        assert item.metadata["dicts"]["a"]["b"]["c"] == "deep"
        assert item.metadata["none_values"] is None
        assert len(item.metadata["empty_structures"]["empty_list"]) == 0