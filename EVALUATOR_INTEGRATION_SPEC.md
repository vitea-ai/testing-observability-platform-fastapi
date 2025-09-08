# Intelligent Content Matcher - Evaluator Service Integration Specification

## Overview
The `intelligent_content_matcher` evaluator uses LLM-based semantic analysis to compare expected content against actual output, with weighted scoring based on metadata priorities.

## Evaluator Configuration

### Basic Information
- **ID**: `intelligent_content_matcher`
- **Name**: Intelligent Content Matcher
- **Type**: `llm_judge`
- **Category**: `quality`

### Configuration Schema
```json
{
  "judge_model": {
    "type": "string",
    "default": "gpt-4",
    "description": "LLM model to use for evaluation"
  },
  "temperature": {
    "type": "number",
    "default": 0.3,
    "minimum": 0.0,
    "maximum": 2.0,
    "description": "Model temperature for consistency"
  },
  "matching_strategy": {
    "type": "string",
    "enum": ["semantic", "exact", "fuzzy"],
    "default": "semantic",
    "description": "Strategy for content matching"
  },
  "weight_mapping": {
    "type": "object",
    "default": {"High": 3, "Medium": 2, "Low": 1},
    "description": "Weight values for meta_weight priorities"
  },
  "confidence_threshold": {
    "type": "number",
    "default": 0.8,
    "minimum": 0.0,
    "maximum": 1.0,
    "description": "Minimum confidence for positive match"
  },
  "include_explanations": {
    "type": "boolean",
    "default": true,
    "description": "Include detailed explanations for each match"
  }
}
```

## Input Data Format

### Expected Input Structure
The evaluator expects experiment data with the following fields:
- `expected_outcome`: Content that should be found in the actual output
- `actual_output`: The generated content to analyze
- `meta_weight`: Priority level ("High", "Medium", "Low")
- `test_id`: Unique identifier for the test case

### Example Input
```json
{
  "test_cases": [
    {
      "test_id": "test_1",
      "expected_outcome": "Referred by PCP for evaluation of palpitations",
      "actual_output": "History of Present Illness - Donald was referred by his primary care physician for evaluation of heart palpitations...",
      "meta_weight": "High"
    }
  ],
  "config": {
    "judge_model": "gpt-4",
    "temperature": 0.3,
    "matching_strategy": "semantic",
    "confidence_threshold": 0.8
  }
}
```

## LLM Evaluation Process

### Prompt Template
```
Task: Determine if the expected content is present in the actual output using semantic analysis.

Expected Content: "{expected_outcome}"
Actual Output: "{actual_output}"

Instructions:
1. Analyze if the expected content appears in the actual output
2. Consider semantic meaning, not just exact text matching
3. Look for equivalent concepts and paraphrased content
4. Assess how completely the expected content is covered

Respond in JSON format:
{
  "match_found": true/false,
  "confidence": 0.0-1.0,
  "coverage": 0.0-1.0,
  "explanation": "brief justification for the decision"
}
```

### Scoring Algorithm
1. **Individual Test Score**: 
   - Base score = confidence × coverage (if match_found, else 0)
   - Weighted score = base_score × weight_value
   
2. **Overall Score**:
   - Total weighted score = sum(weighted_scores)
   - Total possible score = sum(weight_values)
   - Final score = (total_weighted_score / total_possible_score) × 100

## Output Format

### Individual Test Result
```json
{
  "test_id": "test_1",
  "expected_outcome": "Referred by PCP for evaluation...",
  "match_found": true,
  "confidence": 0.95,
  "coverage": 0.90,
  "weight": "High",
  "weight_value": 3,
  "base_score": 0.855,
  "weighted_score": 2.565,
  "explanation": "Found clear reference to PCP referral in the HPI section..."
}
```

### Overall Evaluation Result
```json
{
  "evaluator_id": "intelligent_content_matcher",
  "status": "completed",
  "score": 85.5,
  "total_tests": 9,
  "matches_found": 8,
  "execution_time": 12.3,
  "details": {
    "individual_results": [...],
    "summary": {
      "total_possible_score": 21.0,
      "total_weighted_score": 17.955,
      "high_priority_matches": 3,
      "medium_priority_matches": 4,
      "low_priority_matches": 1,
      "average_confidence": 0.89
    }
  }
}
```

## Integration Points

### API Endpoint
- **POST** `/evaluate/intelligent_content_matcher`
- Follows existing evaluator service patterns
- Uses same authentication and error handling as other evaluators

### Error Handling
- LLM API failures: Retry with exponential backoff
- Invalid JSON responses: Log error and assign 0 score
- Timeout handling: Follow existing evaluator service timeout patterns

### Logging
- Log each LLM call with request/response for debugging
- Track confidence scores and match patterns
- Monitor evaluation performance metrics

## Testing Strategy

### Primary Test Case
- **Dataset**: `samples/kelly_cardiology_experiment.csv`
- **Expected**: 9 test cases with various priority levels
- **Validation**: Confirm semantic matching works for medical documentation

### Success Criteria
1. Evaluator appears in `/api/v1/evaluators` endpoint
2. Configuration schema validates correctly
3. Can process kelly_cardiology_experiment.csv data structure
4. Returns weighted scores respecting High/Medium/Low priorities
5. Provides meaningful explanations for match decisions

## Future Enhancements
- Support for custom prompt templates
- Batch processing optimization
- Caching for repeated evaluations
- Support for different domains beyond healthcare