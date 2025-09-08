"""
Unit tests for dataset schemas and conversation validation.
"""

import pytest
from pydantic import ValidationError

from app.schemas.dataset import (
    ConversationTurn,
    ConversationInput,
    SingleTurnInput,
    DatasetItemBase,
    DatasetCreate,
    CSVUploadResponse
)


class TestConversationSchemas:
    """Test cases for conversation-related schemas."""

    def test_conversation_turn_valid(self):
        """Test valid conversation turn creation."""
        turn = ConversationTurn(
            role="user",
            content="Hello, how are you?",
            metadata={"timestamp": "2023-01-01T00:00:00Z"}
        )
        
        assert turn.role == "user"
        assert turn.content == "Hello, how are you?"
        assert turn.metadata["timestamp"] == "2023-01-01T00:00:00Z"

    def test_conversation_turn_valid_roles(self):
        """Test all valid roles for conversation turns."""
        valid_roles = ["user", "assistant", "system"]
        
        for role in valid_roles:
            turn = ConversationTurn(role=role, content="Test content")
            assert turn.role == role

    def test_conversation_turn_invalid_role(self):
        """Test invalid role raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ConversationTurn(role="invalid_role", content="Test content")
        
        assert "Input should be 'user', 'assistant' or 'system'" in str(exc_info.value)

    def test_conversation_turn_empty_content(self):
        """Test conversation turn with empty content."""
        turn = ConversationTurn(role="user", content="")
        assert turn.content == ""

    def test_conversation_turn_optional_metadata(self):
        """Test conversation turn without metadata."""
        turn = ConversationTurn(role="assistant", content="Response")
        assert turn.metadata == {}

    def test_conversation_input_valid(self):
        """Test valid conversation input creation."""
        turns = [
            ConversationTurn(role="user", content="Hello"),
            ConversationTurn(role="assistant", content="Hi there!")
        ]
        
        conv_input = ConversationInput(
            scenario="Greeting conversation",
            turns=turns
        )
        
        assert conv_input.scenario == "Greeting conversation"
        assert len(conv_input.turns) == 2
        assert conv_input.turns[0].role == "user"

    def test_conversation_input_minimum_turns(self):
        """Test conversation input requires at least one turn."""
        with pytest.raises(ValidationError) as exc_info:
            ConversationInput(scenario="Empty conversation", turns=[])
        
        assert "at least 1 item" in str(exc_info.value)

    def test_conversation_input_multiple_turns(self):
        """Test conversation with multiple turns."""
        turns = [
            ConversationTurn(role="user", content="Hello"),
            ConversationTurn(role="assistant", content="Hi!"),
            ConversationTurn(role="user", content="How are you?"),
            ConversationTurn(role="assistant", content="I'm doing well!")
        ]
        
        conv_input = ConversationInput(scenario="Multi-turn chat", turns=turns)
        assert len(conv_input.turns) == 4


class TestSingleTurnInput:
    """Test cases for single turn input schema."""

    def test_single_turn_input_question(self):
        """Test single turn input with question field."""
        single_input = SingleTurnInput(question="What is AI?")
        assert single_input.question == "What is AI?"
        assert single_input.prompt is None

    def test_single_turn_input_prompt(self):
        """Test single turn input with prompt field."""
        single_input = SingleTurnInput(prompt="Explain machine learning")
        assert single_input.prompt == "Explain machine learning"
        assert single_input.question is None

    def test_single_turn_input_text(self):
        """Test single turn input with text field."""
        single_input = SingleTurnInput(text="Analyze this document")
        assert single_input.text == "Analyze this document"

    def test_single_turn_input_all_fields(self):
        """Test single turn input with all fields."""
        single_input = SingleTurnInput(
            question="What is this?",
            prompt="Please explain",
            text="Some text"
        )
        
        assert single_input.question == "What is this?"
        assert single_input.prompt == "Please explain"
        assert single_input.text == "Some text"

    def test_single_turn_input_extra_fields(self):
        """Test single turn input allows extra fields."""
        # Should allow extra fields due to Config.extra = "allow"
        single_input = SingleTurnInput(
            question="Test?",
            custom_field="custom_value"
        )
        assert single_input.question == "Test?"


class TestDatasetItemBase:
    """Test cases for DatasetItemBase schema."""

    def test_dataset_item_string_input(self):
        """Test dataset item with string input."""
        item = DatasetItemBase(
            input="What is Python?",
            expected_output="A programming language"
        )
        
        assert item.input == "What is Python?"
        assert item.expected_output == "A programming language"

    def test_dataset_item_single_turn_input(self):
        """Test dataset item with SingleTurnInput."""
        single_input = SingleTurnInput(question="What is AI?")
        item = DatasetItemBase(
            input=single_input,
            expected_output="Artificial Intelligence"
        )
        
        assert isinstance(item.input, SingleTurnInput)
        assert item.input.question == "What is AI?"

    def test_dataset_item_conversation_input(self):
        """Test dataset item with ConversationInput."""
        turns = [
            ConversationTurn(role="user", content="Hello"),
            ConversationTurn(role="assistant", content="Hi!")
        ]
        conv_input = ConversationInput(scenario="Greeting", turns=turns)
        
        item = DatasetItemBase(
            input=conv_input,
            expected_outcome="Friendly greeting"
        )
        
        assert isinstance(item.input, ConversationInput)
        assert item.expected_outcome == "Friendly greeting"

    def test_dataset_item_dict_input(self):
        """Test dataset item with dictionary input."""
        dict_input = {
            "scenario": "Test scenario",
            "turns": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ]
        }
        
        item = DatasetItemBase(input=dict_input)
        assert isinstance(item.input, dict)
        assert item.input["scenario"] == "Test scenario"

    def test_dataset_item_with_context(self):
        """Test dataset item with context."""
        item = DatasetItemBase(
            input="Test input",
            context=["background info", "additional context"]
        )
        
        assert len(item.context) == 2
        assert "background info" in item.context

    def test_dataset_item_with_metadata(self):
        """Test dataset item with metadata."""
        metadata = {
            "domain": "healthcare",
            "complexity": "medium",
            "tags": ["medical", "appointment"]
        }
        
        item = DatasetItemBase(
            input="Schedule appointment",
            metadata=metadata
        )
        
        assert item.metadata["domain"] == "healthcare"
        assert item.metadata["complexity"] == "medium"

    def test_dataset_item_with_tags(self):
        """Test dataset item with tags."""
        item = DatasetItemBase(
            input="Test input",
            tags=["test", "validation", "unit"]
        )
        
        assert len(item.tags) == 3
        assert "validation" in item.tags

    def test_dataset_item_optional_fields(self):
        """Test dataset item with only required fields."""
        item = DatasetItemBase(input="Minimal input")
        
        assert item.input == "Minimal input"
        assert item.expected_output is None
        assert item.expected_outcome is None
        assert item.context is None
        assert item.metadata == {}
        assert item.test_id is None
        assert item.tags is None

    def test_dataset_item_both_output_fields(self):
        """Test dataset item with both expected_output and expected_outcome."""
        item = DatasetItemBase(
            input="Test input",
            expected_output="Direct answer",
            expected_outcome="Successful completion"
        )
        
        assert item.expected_output == "Direct answer"
        assert item.expected_outcome == "Successful completion"


class TestDatasetCreate:
    """Test cases for DatasetCreate schema."""

    def test_dataset_create_minimal(self):
        """Test minimal dataset creation."""
        items = [DatasetItemBase(input="Test input")]
        
        dataset = DatasetCreate(
            name="Test Dataset",
            data=items
        )
        
        assert dataset.name == "Test Dataset"
        assert len(dataset.data) == 1
        assert dataset.type == "custom"  # default value
        assert dataset.created_by == "system"  # default value

    def test_dataset_create_full(self):
        """Test full dataset creation with all fields."""
        items = [
            DatasetItemBase(input="Input 1", expected_output="Output 1"),
            DatasetItemBase(input="Input 2", expected_output="Output 2")
        ]
        
        dataset = DatasetCreate(
            name="Full Dataset",
            description="Comprehensive test dataset",
            type="healthcare",
            data=items,
            tags=["test", "healthcare"],
            metadata={"version": "1.0", "source": "manual"},
            created_by="test_user"
        )
        
        assert dataset.name == "Full Dataset"
        assert dataset.description == "Comprehensive test dataset"
        assert dataset.type == "healthcare"
        assert len(dataset.data) == 2
        assert "healthcare" in dataset.tags
        assert dataset.metadata["version"] == "1.0"
        assert dataset.created_by == "test_user"

    def test_dataset_create_validation_empty_name(self):
        """Test validation fails with empty name."""
        items = [DatasetItemBase(input="Test")]
        
        with pytest.raises(ValidationError) as exc_info:
            DatasetCreate(name="", data=items)
        
        assert "at least 1 character" in str(exc_info.value)

    def test_dataset_create_validation_empty_data(self):
        """Test validation fails with empty data."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetCreate(name="Test Dataset", data=[])
        
        assert "at least 1 item" in str(exc_info.value)

    def test_dataset_create_validation_long_name(self):
        """Test validation fails with name too long."""
        items = [DatasetItemBase(input="Test")]
        long_name = "x" * 300  # Exceeds 255 character limit
        
        with pytest.raises(ValidationError) as exc_info:
            DatasetCreate(name=long_name, data=items)
        
        assert "at most 255 characters" in str(exc_info.value)


class TestCSVUploadResponse:
    """Test cases for CSV upload response schema."""

    def test_csv_upload_response_minimal(self):
        """Test minimal CSV upload response."""
        from app.schemas.dataset import DatasetResponse
        from datetime import datetime
        from uuid import uuid4
        
        # Create a mock dataset response
        dataset_dict = {
            "name": "Test Dataset",
            "description": "Test description",
            "type": "custom",
            "data": [{"input": "test", "metadata": {}}],
            "tags": [],
            "metadata": {},
            "id": str(uuid4()),
            "status": "active",
            "record_count": 1,
            "created_by": "system",
            "created_at": datetime.utcnow(),
            "updated_at": None,
            "version": "1.0.0"
        }
        
        dataset_response = DatasetResponse(**dataset_dict)
        
        response = CSVUploadResponse(
            message="Upload successful",
            dataset=dataset_response,
            rows_processed=1
        )
        
        assert response.message == "Upload successful"
        assert response.rows_processed == 1
        assert response.warnings == []  # default empty list

    def test_csv_upload_response_with_warnings(self):
        """Test CSV upload response with warnings."""
        from app.schemas.dataset import DatasetResponse
        from datetime import datetime
        from uuid import uuid4
        
        dataset_dict = {
            "name": "Test Dataset",
            "description": "Test description", 
            "type": "custom",
            "data": [{"input": "test", "metadata": {}}],
            "tags": [],
            "metadata": {},
            "id": str(uuid4()),
            "status": "active",
            "record_count": 1,
            "created_by": "system",
            "created_at": datetime.utcnow(),
            "updated_at": None,
            "version": "1.0.0"
        }
        
        dataset_response = DatasetResponse(**dataset_dict)
        
        warnings = [
            "Row 5: Missing expected_output field",
            "Row 10: Invalid role 'moderator', defaulting to 'user'"
        ]
        
        response = CSVUploadResponse(
            message="Upload completed with warnings",
            dataset=dataset_response,
            rows_processed=15,
            warnings=warnings
        )
        
        assert response.message == "Upload completed with warnings"
        assert response.rows_processed == 15
        assert len(response.warnings) == 2
        assert "Row 5" in response.warnings[0]


class TestSchemaIntegration:
    """Integration tests for schema combinations."""

    def test_conversation_to_dataset_item_serialization(self):
        """Test serializing conversation input to dataset item."""
        turns = [
            ConversationTurn(role="user", content="Hello", metadata={"timestamp": "2023-01-01"}),
            ConversationTurn(role="assistant", content="Hi there!")
        ]
        
        conv_input = ConversationInput(scenario="Customer service", turns=turns)
        
        item = DatasetItemBase(
            input=conv_input,
            expected_outcome="Helpful response",
            tags=["customer_service", "greeting"],
            metadata={"priority": "high"}
        )
        
        # Test serialization to dict
        item_dict = item.model_dump()
        
        assert item_dict["input"]["scenario"] == "Customer service"
        assert len(item_dict["input"]["turns"]) == 2
        assert item_dict["input"]["turns"][0]["role"] == "user"
        assert item_dict["expected_outcome"] == "Helpful response"
        assert "customer_service" in item_dict["tags"]

    def test_mixed_input_types_in_dataset(self):
        """Test dataset with mixed input types."""
        items = [
            DatasetItemBase(input="Simple string input"),
            DatasetItemBase(
                input=SingleTurnInput(question="What is AI?"),
                expected_output="Artificial Intelligence"
            ),
            DatasetItemBase(
                input=ConversationInput(
                    scenario="Multi-turn",
                    turns=[ConversationTurn(role="user", content="Hello")]
                ),
                expected_outcome="Greeting response"
            ),
            DatasetItemBase(
                input={"custom": "dictionary", "format": "flexible"}
            )
        ]
        
        dataset = DatasetCreate(
            name="Mixed Format Dataset",
            data=items
        )
        
        assert len(dataset.data) == 4
        assert isinstance(dataset.data[0].input, str)
        assert isinstance(dataset.data[1].input, SingleTurnInput)
        assert isinstance(dataset.data[2].input, ConversationInput)
        assert isinstance(dataset.data[3].input, dict)

    def test_complex_metadata_handling(self):
        """Test complex metadata structures."""
        complex_metadata = {
            "domain": "healthcare",
            "sub_domains": ["appointments", "billing", "records"],
            "difficulty_metrics": {
                "lexical_complexity": 0.7,
                "semantic_difficulty": 0.8,
                "conversation_length": 5
            },
            "evaluation_criteria": [
                {"metric": "accuracy", "threshold": 0.9},
                {"metric": "relevance", "threshold": 0.8}
            ]
        }
        
        item = DatasetItemBase(
            input="Complex healthcare query",
            metadata=complex_metadata
        )
        
        # Verify complex metadata is preserved
        assert item.metadata["domain"] == "healthcare"
        assert len(item.metadata["sub_domains"]) == 3
        assert item.metadata["difficulty_metrics"]["lexical_complexity"] == 0.7
        assert len(item.metadata["evaluation_criteria"]) == 2