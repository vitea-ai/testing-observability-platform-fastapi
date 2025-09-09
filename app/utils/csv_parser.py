"""
Production-ready CSV parser with streaming and chunking support.
"""

import io
import csv
import json
import asyncio
from typing import List, Dict, Any, AsyncIterator, Tuple, Optional
from collections import defaultdict
import pandas as pd
import aiofiles
import tempfile
from pathlib import Path

from fastapi import UploadFile, HTTPException
from pydantic import ValidationError

from app.schemas.dataset import (
    DatasetItemBase,
    ConversationTurn,
    ConversationInput,
    SingleTurnInput
)
from app.core.logging import logger


class CSVParser:
    """Production-ready CSV parser with streaming and format detection."""
    
    # Configuration
    CHUNK_SIZE = 1000  # Process 1000 rows at a time
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max
    SAMPLE_SIZE = 100  # Rows to sample for format detection
    MAX_ERRORS = 10  # Maximum validation errors before failing
    
    def __init__(self):
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.processed_count = 0
        
    async def parse_csv_file(
        self, 
        file: UploadFile,
        validate_only: bool = False
    ) -> Tuple[List[DatasetItemBase], List[str], Dict[str, Any]]:
        """
        Parse CSV file with streaming and chunking.
        
        Args:
            file: Uploaded CSV file
            validate_only: If True, only validate without full parsing
            
        Returns:
            Tuple of (parsed items, warnings, metadata)
        """
        # Reset state
        self.warnings = []
        self.errors = []
        self.processed_count = 0
        
        # Create temporary file for streaming
        temp_file_path = None
        try:
            # Stream file to temporary location
            temp_file_path = await self._stream_to_temp_file(file)
            
            # Detect format from sample
            format_info = await self._detect_format(temp_file_path)
            
            # Parse based on detected format
            if format_info['type'] == 'conversation':
                items = await self._parse_conversation_format(temp_file_path, validate_only)
            else:
                items = await self._parse_simple_format(temp_file_path, validate_only)
            
            metadata = {
                'format_type': format_info['type'],
                'total_rows': format_info['total_rows'],
                'detected_columns': format_info['columns'],
                'processed_items': len(items),
                'warnings_count': len(self.warnings)
            }
            
            return items, self.warnings, metadata
            
        finally:
            # Clean up temp file
            if temp_file_path and Path(temp_file_path).exists():
                Path(temp_file_path).unlink()
    
    async def _stream_to_temp_file(self, file: UploadFile) -> str:
        """Stream uploaded file to temporary location with size validation."""
        total_size = 0
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp:
            temp_path = tmp.name
            
            # Stream in chunks
            while chunk := await file.read(8192):  # 8KB chunks
                total_size += len(chunk)
                
                if total_size > self.MAX_FILE_SIZE:
                    Path(temp_path).unlink()
                    raise HTTPException(
                        status_code=413,
                        detail=f"File size exceeds maximum of {self.MAX_FILE_SIZE // (1024*1024)}MB"
                    )
                
                tmp.write(chunk)
        
        logger.info(f"Streamed {total_size} bytes to temporary file")
        return temp_path
    
    async def _detect_format(self, file_path: str) -> Dict[str, Any]:
        """Detect CSV format from sample rows."""
        # Read sample with pandas for quick analysis
        try:
            df_sample = pd.read_csv(file_path, nrows=self.SAMPLE_SIZE)
            total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read CSV file: {str(e)}"
            )
        
        columns = list(df_sample.columns)
        format_type = 'simple'  # Default
        
        # Detect conversation format
        if 'role' in columns and 'content' in columns:
            format_type = 'conversation'
            # Check if we have valid roles
            if not df_sample['role'].isin(['user', 'assistant', 'system']).any():
                self.warnings.append("'role' column found but contains non-standard values")
        
        # Detect structured format with specific columns
        elif {'input', 'expected_output'}.issubset(columns):
            format_type = 'structured'
        
        # Detect flexible input format
        elif any(col in columns for col in ['prompt', 'question', 'text', 'input_prompt']):
            format_type = 'flexible'
        
        return {
            'type': format_type,
            'columns': columns,
            'total_rows': total_rows,
            'sample_rows': len(df_sample)
        }
    
    async def _parse_conversation_format(
        self, 
        file_path: str,
        validate_only: bool = False
    ) -> List[DatasetItemBase]:
        """Parse conversation format CSV with chunking."""
        items = []
        conversations = defaultdict(lambda: {
            'scenario': '',
            'turns': [],
            'expected_outcome': '',
            'metadata': {},
            'tags': set()
        })
        
        # Process in chunks using pandas
        chunk_iterator = pd.read_csv(file_path, chunksize=self.CHUNK_SIZE)
        
        for chunk_num, chunk_df in enumerate(chunk_iterator):
            logger.info(f"Processing chunk {chunk_num + 1} with {len(chunk_df)} rows")
            
            for _, row in chunk_df.iterrows():
                self.processed_count += 1
                
                # Get grouping key
                group_key = row.get('test_id') or row.get('scenario') or f"conv_{self.processed_count}"
                
                # Build conversation
                conv = conversations[group_key]
                
                if row.get('scenario'):
                    conv['scenario'] = row['scenario']
                
                # Add turn
                role = str(row.get('role', 'user')).lower()
                content = str(row.get('content', ''))
                
                if content:
                    turn_metadata = {k: v for k, v in row.items() if k.startswith('meta_')}
                    conv['turns'].append(
                        ConversationTurn(
                            role=role if role in ['user', 'assistant', 'system'] else 'user',
                            content=content,
                            metadata=turn_metadata if turn_metadata else None
                        )
                    )
                
                # Update metadata
                if row.get('expected_outcome'):
                    conv['expected_outcome'] = row['expected_outcome']
                
                if row.get('tags'):
                    conv['tags'].update(t.strip() for t in str(row['tags']).split(','))
            
            # Validate chunk if we have enough conversations
            if len(conversations) >= self.CHUNK_SIZE:
                if validate_only:
                    self._validate_conversations(dict(list(conversations.items())[:self.CHUNK_SIZE]))
                else:
                    items.extend(self._convert_conversations_to_items(
                        dict(list(conversations.items())[:self.CHUNK_SIZE])
                    ))
                
                # Clear processed conversations
                for key in list(conversations.keys())[:self.CHUNK_SIZE]:
                    del conversations[key]
        
        # Process remaining conversations
        if not validate_only:
            items.extend(self._convert_conversations_to_items(conversations))
        
        return items
    
    async def _parse_simple_format(
        self,
        file_path: str,
        validate_only: bool = False
    ) -> List[DatasetItemBase]:
        """Parse simple/structured format CSV with chunking."""
        items = []
        validation_errors = 0
        
        # Process in chunks using pandas
        chunk_iterator = pd.read_csv(file_path, chunksize=self.CHUNK_SIZE)
        
        for chunk_num, chunk_df in enumerate(chunk_iterator):
            logger.info(f"Processing chunk {chunk_num + 1} with {len(chunk_df)} rows")
            
            chunk_items = []
            
            for idx, row in chunk_df.iterrows():
                self.processed_count += 1
                
                try:
                    item = self._parse_single_row(row.to_dict(), self.processed_count)
                    if item:
                        chunk_items.append(item)
                    else:
                        logger.debug(f"Row {self.processed_count} returned None from _parse_single_row")
                except ValidationError as e:
                    validation_errors += 1
                    self.errors.append(f"Row {self.processed_count}: {str(e)}")
                    
                    if validation_errors >= self.MAX_ERRORS:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Too many validation errors (>{self.MAX_ERRORS}). First errors: {self.errors[:5]}"
                        )
            
            # Batch validate if needed
            logger.info(f"Chunk {chunk_num + 1}: {len(chunk_items)} items parsed, validate_only={validate_only}")
            if not validate_only and chunk_items:
                validated_items = self._batch_validate(chunk_items)
                logger.info(f"Chunk {chunk_num + 1}: {len(validated_items)} items after validation")
                items.extend(validated_items)
            elif validate_only:
                logger.info(f"Skipping batch validation due to validate_only=True")
        
        return items
    
    def _parse_single_row(self, row: Dict[str, Any], row_num: int) -> Optional[DatasetItemBase]:
        """Parse a single row to DatasetItemBase."""
        try:
            import pandas as pd
            # Clean row - convert NaN to None
            clean_row = {}
            for k, v in row.items():
                if pd.isna(v):
                    clean_row[k] = None
                else:
                    clean_row[k] = v
        except ImportError:
            # Fallback if pandas is not available
            clean_row = {}
            for k, v in row.items():
                if v is None or (isinstance(v, float) and str(v) == 'nan'):
                    clean_row[k] = None
                else:
                    clean_row[k] = v
        
        # Find input field
        input_value = (
            clean_row.get('input') or
            clean_row.get('prompt') or
            clean_row.get('question') or
            clean_row.get('input_prompt') or
            clean_row.get('text')
        )
        
        if not input_value or pd.isna(input_value):
            self.warnings.append(f"Row {row_num}: No input field found")
            return None
        
        # Parse context
        context = []
        if clean_row.get('context'):
            try:
                context = json.loads(clean_row['context']) if isinstance(clean_row['context'], str) else clean_row['context']
                if not isinstance(context, list):
                    context = [str(context)]
            except (json.JSONDecodeError, TypeError):
                context = [c.strip() for c in str(clean_row['context']).split(',') if c.strip()]
        
        # Parse tags
        tags = []
        if clean_row.get('tags'):
            tags = [t.strip() for t in str(clean_row['tags']).split(',') if t.strip()]
        
        # Extract metadata - include conversation_id and other relevant fields
        metadata = {k: v for k, v in clean_row.items() if k.startswith('meta_') and v is not None}
        
        # Add conversation_id to metadata if present
        if clean_row.get('conversation_id'):
            metadata['conversation_id'] = clean_row['conversation_id']
        
        # Add output field to metadata for multi-conversation format
        if clean_row.get('output'):
            metadata['output'] = clean_row['output']
        
        # Include any existing metadata field
        if clean_row.get('metadata'):
            try:
                # Try to parse as JSON if it's a string
                if isinstance(clean_row['metadata'], str):
                    parsed_meta = json.loads(clean_row['metadata'])
                    if isinstance(parsed_meta, dict):
                        metadata.update(parsed_meta)
                    else:
                        metadata['raw_metadata'] = clean_row['metadata']
                else:
                    metadata['raw_metadata'] = str(clean_row['metadata'])
            except (json.JSONDecodeError, TypeError):
                metadata['raw_metadata'] = str(clean_row['metadata'])
        
        return DatasetItemBase(
            input=input_value,
            expected_output=clean_row.get('expected_output'),
            expected_outcome=clean_row.get('expected_outcome'),
            context=context or None,
            test_id=clean_row.get('test_id') or clean_row.get('conversation_id'),
            tags=tags or None,
            metadata=metadata or None
        )
    
    def _convert_conversations_to_items(
        self, 
        conversations: Dict[str, Dict]
    ) -> List[DatasetItemBase]:
        """Convert conversation dictionaries to DatasetItemBase objects."""
        items = []
        
        for conv_id, conv_data in conversations.items():
            if not conv_data['turns']:
                self.warnings.append(f"Conversation '{conv_id}' has no turns")
                continue
            
            try:
                conv_input = ConversationInput(
                    scenario=conv_data['scenario'] or f"Conversation {conv_id}",
                    turns=conv_data['turns']
                )
                
                item = DatasetItemBase(
                    input=conv_input.model_dump(),
                    expected_outcome=conv_data['expected_outcome'] or None,
                    tags=list(conv_data['tags']) if conv_data['tags'] else None,
                    metadata=conv_data['metadata'] if conv_data['metadata'] else None
                )
                items.append(item)
            except ValidationError as e:
                self.errors.append(f"Conversation '{conv_id}': {str(e)}")
        
        return items
    
    def _validate_conversations(self, conversations: Dict[str, Dict]) -> None:
        """Validate conversation structures."""
        for conv_id, conv_data in conversations.items():
            if not conv_data['turns']:
                self.warnings.append(f"Conversation '{conv_id}' has no valid turns")
            
            # Validate turn sequence
            roles = [turn.role for turn in conv_data['turns']]
            if roles and roles[0] == 'assistant':
                self.warnings.append(f"Conversation '{conv_id}' starts with assistant role")
    
    def _batch_validate(self, items: List[DatasetItemBase]) -> List[DatasetItemBase]:
        """Batch validate items for efficiency."""
        validated = []
        
        for item in items:
            try:
                # Pydantic validation happens automatically
                validated.append(item)
            except ValidationError as e:
                self.errors.append(f"Validation error: {str(e)}")
        
        return validated
    
    async def parse_experiment_results(
        self,
        file: UploadFile,
        validate_only: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
        """
        Parse CSV file for experiment results import.
        
        Args:
            file: Uploaded CSV file
            validate_only: If True, only validate without full parsing
            
        Returns:
            Tuple of (parsed results, warnings, metadata)
        """
        # Reset state
        self.warnings = []
        self.errors = []
        self.processed_count = 0
        
        # Create temporary file for streaming
        temp_file_path = None
        try:
            # Stream file to temporary location
            temp_file_path = await self._stream_to_temp_file(file)
            
            # Parse experiment results
            results = await self._parse_experiment_format(temp_file_path, validate_only)
            
            # Count total rows
            total_rows = sum(1 for _ in open(temp_file_path)) - 1  # Subtract header
            
            metadata = {
                'format_type': 'experiment_results',
                'total_rows': total_rows,
                'processed_items': len(results),
                'warnings_count': len(self.warnings)
            }
            
            return results, self.warnings, metadata
            
        finally:
            # Clean up temp file
            if temp_file_path and Path(temp_file_path).exists():
                Path(temp_file_path).unlink()
    
    async def _parse_experiment_format(
        self,
        file_path: str,
        validate_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Parse experiment results format CSV with chunking."""
        results = []
        validation_errors = 0
        
        # Process in chunks using pandas
        chunk_iterator = pd.read_csv(file_path, chunksize=self.CHUNK_SIZE)
        
        for chunk_num, chunk_df in enumerate(chunk_iterator):
            logger.info(f"Processing experiment chunk {chunk_num + 1} with {len(chunk_df)} rows")
            
            for idx, row in chunk_df.iterrows():
                self.processed_count += 1
                
                try:
                    result = self._parse_experiment_row(row.to_dict(), self.processed_count)
                    if result:
                        results.append(result)
                except Exception as e:
                    validation_errors += 1
                    self.errors.append(f"Row {self.processed_count}: {str(e)}")
                    
                    if validation_errors >= self.MAX_ERRORS:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Too many validation errors (>{self.MAX_ERRORS}). First errors: {self.errors[:5]}"
                        )
            
            # Early return if just validating  
            if validate_only and len(results) >= self.SAMPLE_SIZE:
                return results[:self.SAMPLE_SIZE]
        
        return results
    
    def _parse_experiment_row(self, row: Dict[str, Any], row_num: int) -> Optional[Dict[str, Any]]:
        """Parse a single experiment result row."""
        # Find input field
        input_value = (
            row.get('input') or
            row.get('input_prompt') or
            row.get('prompt') or
            row.get('question') or
            row.get('text')
        )
        
        if not input_value:
            # Provide helpful error message with available columns
            available_columns = [col for col in row.keys() if row.get(col)]
            self.warnings.append(
                f"Row {row_num}: No input field found. Expected columns with names like: 'input', 'prompt', 'question', or 'text'. "
                f"Found columns: {', '.join(available_columns[:7])}"
            )
            return None
        
        # Extract metadata from meta_* fields
        metadata = {}
        for key, value in row.items():
            if key and key.startswith('meta_'):
                metadata[key] = value
        
        # Parse JSON fields if they exist
        retrieval_context = []
        if row.get('retrieval_context'):
            try:
                retrieval_context = json.loads(row['retrieval_context'])
            except:
                pass
        
        tools_called = []
        if row.get('tools_called'):
            try:
                tools_called = json.loads(row['tools_called'])
            except:
                pass
        
        # Parse context
        context = []
        if row.get('context'):
            try:
                context = json.loads(row['context']) if isinstance(row['context'], str) else row['context']
                if not isinstance(context, list):
                    context = [str(context)]
            except (json.JSONDecodeError, TypeError):
                context = [str(row['context'])] if row['context'] else []
        
        # Create result item - intelligently find expected and actual columns
        # Find any column containing "expected"
        expected_value = ''
        for col_name, col_value in row.items():
            if col_name and 'expected' in col_name.lower() and col_value:
                expected_value = col_value
                break
        
        # Find any column containing "actual" 
        actual_value = ''
        for col_name, col_value in row.items():
            if col_name and 'actual' in col_name.lower() and col_value:
                actual_value = col_value
                break
        
        # Fallback to 'output' if no 'actual' column found
        if not actual_value:
            actual_value = row.get('output', '')
        
        result = {
            "test_id": row.get('test_case_id') or row.get('test_id') or f"test_{row_num}",
            "test_case_type": row.get('test_case_type', 'single_turn'),
            "input": input_value,
            "expected_output": expected_value,
            "actual_output": actual_value,
            "context": context,
            "retrieval_context": retrieval_context,
            "tools_called": tools_called,
            "status": row.get('status', 'completed'),
            "execution_time": float(row.get('latency_ms', 0)) / 1000 if row.get('latency_ms') else (
                float(row.get('execution_time', 0)) if row.get('execution_time') else 0
            ),
            "token_usage": {
                "input": int(row.get('token_usage_input', 0)) if row.get('token_usage_input') else 0,
                "output": int(row.get('token_usage_output', 0)) if row.get('token_usage_output') else 0
            } if row.get('token_usage_input') or row.get('token_usage_output') else None,
            "error": row.get('error'),
            "meta_data": metadata
        }
        
        return result


# Singleton instance for reuse
csv_parser = CSVParser()