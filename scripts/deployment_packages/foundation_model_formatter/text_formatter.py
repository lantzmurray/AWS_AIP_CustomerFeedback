#!/usr/bin/env python3
"""
Text Data Formatter

Specialized formatter for text review data that integrates with Phase 2 text processing
output and supports multiple foundation model input formats including JSONL, Parquet,
and Conversation formats.
"""

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import re
import boto3
from typing import Dict, List, Any, Optional, Union
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class TextFormatter:
    """
    Specialized formatter for text review data with integration to Phase 2 processing.
    """
    
    def __init__(self):
        """Initialize the text formatter."""
        self.s3_client = boto3.client('s3')
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load text formatting templates."""
        return {
            "claude_analysis": """You are analyzing customer feedback to generate actionable business insights.

CUSTOMER REVIEW:
{original_text}

ANALYSIS DATA:
- Entities: {entities}
- Sentiment: {sentiment} (confidence: {sentiment_score})
- Key Phrases: {key_phrases}

METADATA:
- Product ID: {product_id}
- Customer ID: {customer_id}
- Review Date: {review_date}
- Quality Score: {quality_score}

Please provide:
1. A summary of the customer's main points
2. Key insights for business improvement
3. Recommended actions
4. Sentiment analysis explanation
5. Priority level for follow-up (High/Medium/Low)

Format your response as structured JSON with the following keys:
summary, insights, recommendations, sentiment_analysis, priority_level""",
            
            "titan_analysis": """Analyze this customer review and extract key business insights:

Review: {original_text}

Sentiment: {sentiment}
Key Topics: {key_phrases}
Entities: {entities}

Provide analysis in JSON format with: summary, sentiment_details, key_insights, action_items.""",
            
            "training_prompt": """Customer review: {original_text}

Sentiment: {sentiment}
Key topics: {key_phrases}""",
            
            "training_completion": """{sentiment_analysis} with {priority_level} priority. Key insights: {insights}. Recommended actions: {recommendations}."""
        }
    
    def format_for_claude(self, processed_text: Dict[str, Any], 
                         format_type: str = "conversation") -> Dict[str, Any]:
        """
        Format processed text data for Claude models.
        
        Args:
            processed_text: Processed text data from Phase 2
            format_type: Output format (conversation, jsonl, parquet)
            
        Returns:
            Formatted data for Claude
        """
        if format_type == "conversation":
            return self._format_claude_conversation(processed_text)
        elif format_type == "jsonl":
            return self._format_jsonl(processed_text, "claude")
        elif format_type == "parquet":
            return self._format_parquet(processed_text, "claude")
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def format_for_titan(self, processed_text: Dict[str, Any],
                        format_type: str = "json") -> Dict[str, Any]:
        """
        Format processed text data for Titan models.
        
        Args:
            processed_text: Processed text data from Phase 2
            format_type: Output format (json, jsonl, parquet)
            
        Returns:
            Formatted data for Titan
        """
        if format_type == "json":
            return self._format_titan_json(processed_text)
        elif format_type == "jsonl":
            return self._format_jsonl(processed_text, "titan")
        elif format_type == "parquet":
            return self._format_parquet(processed_text, "titan")
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def format_for_training(self, processed_text: Dict[str, Any],
                           output_format: str = "jsonl") -> Dict[str, Any]:
        """
        Format processed text data for model training.
        
        Args:
            processed_text: Processed text data from Phase 2
            output_format: Output format (jsonl, parquet)
            
        Returns:
            Training data format
        """
        if output_format == "jsonl":
            return self._format_training_jsonl(processed_text)
        elif output_format == "parquet":
            return self._format_training_parquet(processed_text)
        else:
            raise ValueError(f"Unsupported training format: {output_format}")
    
    def _format_claude_conversation(self, processed_text: Dict[str, Any]) -> Dict[str, Any]:
        """Format text data as Claude conversation."""
        
        # Extract and process data
        original_text = processed_text.get("original_text", "")
        entities = processed_text.get("entities", [])
        sentiment = processed_text.get("sentiment", {})
        key_phrases = processed_text.get("key_phrases", [])
        metadata = processed_text.get("metadata", {})
        
        # Format entities for display
        entities_text = json.dumps(entities, indent=2)
        
        # Format sentiment
        sentiment_text = sentiment.get("Sentiment", "Unknown")
        sentiment_score = sentiment.get("Score", 0.0)
        
        # Format key phrases
        key_phrases_text = ", ".join([phrase.get("Text", "") for phrase in key_phrases])
        
        # Extract metadata
        product_id = metadata.get("product_id", "N/A")
        customer_id = metadata.get("customer_id", "N/A")
        review_date = metadata.get("review_date", "N/A")
        quality_score = metadata.get("quality_score", 0.0)
        
        # Create conversation prompt
        prompt = self.templates["claude_analysis"].format(
            original_text=original_text,
            entities=entities_text,
            sentiment=sentiment_text,
            sentiment_score=sentiment_score,
            key_phrases=key_phrases_text,
            product_id=product_id,
            customer_id=customer_id,
            review_date=review_date,
            quality_score=quality_score
        )
        
        # Create Claude conversation format
        conversation = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.999,
            "top_k": 250
        }
        
        # Add metadata
        conversation["metadata"] = {
            "data_type": "text",
            "model": "claude-v2",
            "format_timestamp": datetime.now().isoformat(),
            "source_data_id": metadata.get("id", customer_id),
            "original_sentiment": sentiment_text,
            "quality_score": quality_score
        }
        
        return conversation
    
    def _format_titan_json(self, processed_text: Dict[str, Any]) -> Dict[str, Any]:
        """Format text data for Titan Text model."""
        
        original_text = processed_text.get("original_text", "")
        sentiment = processed_text.get("sentiment", {})
        key_phrases = processed_text.get("key_phrases", [])
        entities = processed_text.get("entities", [])
        
        # Format for Titan
        sentiment_text = sentiment.get("Sentiment", "Unknown")
        key_phrases_text = ", ".join([phrase.get("Text", "") for phrase in key_phrases])
        entities_text = ", ".join([entity.get("Text", "") for entity in entities])
        
        # Create Titan prompt
        prompt = self.templates["titan_analysis"].format(
            original_text=original_text,
            sentiment=sentiment_text,
            key_phrases=key_phrases_text,
            entities=entities_text
        )
        
        titan_request = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 2000,
                "temperature": 0.7,
                "topP": 0.999,
                "stopSequences": []
            }
        }
        
        # Add metadata
        metadata = processed_text.get("metadata", {})
        titan_request["metadata"] = {
            "data_type": "text",
            "model": "titan-text-express-v1",
            "format_timestamp": datetime.now().isoformat(),
            "source_data_id": metadata.get("id", metadata.get("customer_id", "unknown")),
            "original_sentiment": sentiment_text,
            "quality_score": metadata.get("quality_score", 0.0)
        }
        
        return titan_request
    
    def _format_jsonl(self, processed_text: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Format text data as JSONL for training."""
        
        original_text = processed_text.get("original_text", "")
        sentiment = processed_text.get("sentiment", {})
        key_phrases = processed_text.get("key_phrases", [])
        metadata = processed_text.get("metadata", {})
        
        # Create training prompt
        key_phrases_text = ", ".join([phrase.get("Text", "") for phrase in key_phrases])
        prompt = self.templates["training_prompt"].format(
            original_text=original_text,
            sentiment=sentiment.get("Sentiment", "Unknown"),
            key_phrases=key_phrases_text
        )
        
        # Generate completion (would normally come from model)
        completion = self.templates["training_completion"].format(
            sentiment_analysis=sentiment.get("Sentiment", "Unknown").lower(),
            priority_level="Medium",  # Would be determined by analysis
            insights="Customer satisfaction and product quality",
            recommendations="Monitor customer feedback and improve product features"
        )
        
        jsonl_record = {
            "prompt": prompt,
            "completion": completion,
            "data_type": "text",
            "model": model,
            "quality_score": metadata.get("quality_score", 0.0),
            "customer_id": metadata.get("customer_id", ""),
            "product_id": metadata.get("product_id", ""),
            "sentiment": sentiment.get("Sentiment", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        return {"jsonl_line": json.dumps(jsonl_record)}
    
    def _format_parquet(self, processed_text: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Format text data as Parquet for analytics."""
        
        original_text = processed_text.get("original_text", "")
        sentiment = processed_text.get("sentiment", {})
        key_phrases = processed_text.get("key_phrases", [])
        entities = processed_text.get("entities", [])
        metadata = processed_text.get("metadata", {})
        
        # Extract key information
        key_phrases_text = ", ".join([phrase.get("Text", "") for phrase in key_phrases])
        entities_text = ", ".join([entity.get("Text", "") for phrase in entities])
        
        # Create DataFrame
        df_data = {
            "original_text": [original_text],
            "sentiment": [sentiment.get("Sentiment", "")],
            "sentiment_score": [sentiment.get("Score", 0.0)],
            "key_phrases": [key_phrases_text],
            "entities": [entities_text],
            "data_type": ["text"],
            "model": [model],
            "quality_score": [metadata.get("quality_score", 0.0)],
            "customer_id": [metadata.get("customer_id", "")],
            "product_id": [metadata.get("product_id", "")],
            "review_date": [metadata.get("review_date", "")],
            "timestamp": [datetime.now()]
        }
        
        df = pd.DataFrame(df_data)
        
        # Convert to Parquet
        table = pa.Table.from_pandas(df)
        parquet_buffer = pa.BufferOutputStream()
        pq.write_table(table, parquet_buffer)
        
        return {
            "parquet_data": parquet_buffer.getvalue().to_pybytes(),
            "schema": str(table.schema),
            "row_count": len(df)
        }
    
    def _format_training_jsonl(self, processed_text: Dict[str, Any]) -> Dict[str, Any]:
        """Format text data specifically for training."""
        
        original_text = processed_text.get("original_text", "")
        sentiment = processed_text.get("sentiment", {})
        key_phrases = processed_text.get("key_phrases", [])
        metadata = processed_text.get("metadata", {})
        
        # Create fine-tuning format
        training_record = {
            "instruction": "Analyze this customer review and provide business insights.",
            "input": original_text,
            "output": f"""Sentiment: {sentiment.get('Sentiment', 'Unknown')}
Key Topics: {', '.join([phrase.get('Text', '') for phrase in key_phrases])}
Quality Score: {metadata.get('quality_score', 0.0)}""",
            "data_type": "text_review",
            "quality_score": metadata.get("quality_score", 0.0)
        }
        
        return {"jsonl_line": json.dumps(training_record)}
    
    def _format_training_parquet(self, processed_text: Dict[str, Any]) -> Dict[str, Any]:
        """Format text data as Parquet for training."""
        
        original_text = processed_text.get("original_text", "")
        sentiment = processed_text.get("sentiment", {})
        key_phrases = processed_text.get("key_phrases", [])
        entities = processed_text.get("entities", [])
        metadata = processed_text.get("metadata", {})
        
        # Create training DataFrame
        df_data = {
            "instruction": ["Analyze this customer review and provide business insights."],
            "input": [original_text],
            "output": [f"""Sentiment: {sentiment.get('Sentiment', 'Unknown')}
Key Topics: {', '.join([phrase.get('Text', '') for phrase in key_phrases])}
Quality Score: {metadata.get('quality_score', 0.0)}"""],
            "data_type": ["text_review"],
            "quality_score": [metadata.get("quality_score", 0.0)],
            "customer_id": [metadata.get("customer_id", "")],
            "product_id": [metadata.get("product_id", "")],
            "timestamp": [datetime.now()]
        }
        
        df = pd.DataFrame(df_data)
        
        # Convert to Parquet
        table = pa.Table.from_pandas(df)
        parquet_buffer = pa.BufferOutputStream()
        pq.write_table(table, parquet_buffer)
        
        return {
            "parquet_data": parquet_buffer.getvalue().to_pybytes(),
            "schema": str(table.schema),
            "row_count": len(df)
        }
    
    def enhance_with_context(self, processed_text: Dict[str, Any], 
                           context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance processed text with additional context.
        
        Args:
            processed_text: Original processed text data
            context_data: Additional context to add
            
        Returns:
            Enhanced text data
        """
        enhanced_data = processed_text.copy()
        
        # Add customer history context
        if "customer_history" in context_data:
            enhanced_data["customer_history"] = context_data["customer_history"]
        
        # Add product context
        if "product_info" in context_data:
            enhanced_data["product_context"] = context_data["product_info"]
        
        # Add business context
        if "business_context" in context_data:
            enhanced_data["business_context"] = context_data["business_context"]
        
        # Add temporal context
        if "temporal_context" in context_data:
            enhanced_data["temporal_context"] = context_data["temporal_context"]
        
        return enhanced_data
    
    def optimize_tokenization(self, formatted_data: Dict[str, Any], 
                           target_tokens: int = 4000) -> Dict[str, Any]:
        """
        Optimize formatted data for token efficiency.
        
        Args:
            formatted_data: Already formatted data
            target_tokens: Target token count
            
        Returns:
            Token-optimized data
        """
        # Estimate current token count (rough approximation)
        text_content = json.dumps(formatted_data)
        current_tokens = len(text_content.split()) * 1.3
        
        if current_tokens <= target_tokens:
            return formatted_data
        
        # If too long, truncate or summarize
        optimization_ratio = target_tokens / current_tokens
        
        # For conversation format, truncate the prompt
        if "messages" in formatted_data:
            for message in formatted_data["messages"]:
                if "content" in message and isinstance(message["content"], str):
                    original_length = len(message["content"])
                    target_length = int(original_length * optimization_ratio)
                    message["content"] = message["content"][:target_length] + "..."
        
        # For other formats, truncate text fields
        elif "inputText" in formatted_data:
            original_length = len(formatted_data["inputText"])
            target_length = int(original_length * optimization_ratio)
            formatted_data["inputText"] = formatted_data["inputText"][:target_length] + "..."
        
        # Add optimization metadata
        formatted_data["token_optimization"] = {
            "original_tokens": current_tokens,
            "target_tokens": target_tokens,
            "optimization_ratio": optimization_ratio,
            "optimized": True
        }
        
        return formatted_data
    
    def validate_text_data(self, processed_text: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate processed text data for formatting requirements.
        
        Args:
            processed_text: Processed text data to validate
            
        Returns:
            Validation result
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "quality_score": 0.0
        }
        
        # Check required fields
        required_fields = ["original_text", "sentiment", "key_phrases", "metadata"]
        for field in required_fields:
            if field not in processed_text:
                validation_result["valid"] = False
                validation_result["issues"].append(f"Missing required field: {field}")
        
        # Check text quality
        original_text = processed_text.get("original_text", "")
        if len(original_text) < 10:
            validation_result["warnings"].append("Text is very short (< 10 characters)")
        elif len(original_text) > 10000:
            validation_result["warnings"].append("Text is very long (> 10,000 characters)")
        
        # Check sentiment data
        sentiment = processed_text.get("sentiment", {})
        if "Sentiment" not in sentiment:
            validation_result["issues"].append("Missing sentiment classification")
        
        # Calculate overall quality score
        metadata = processed_text.get("metadata", {})
        base_quality = metadata.get("quality_score", 0.0)
        
        # Adjust quality based on validation results
        quality_adjustment = 0.0
        if validation_result["issues"]:
            quality_adjustment -= len(validation_result["issues"]) * 0.2
        if validation_result["warnings"]:
            quality_adjustment -= len(validation_result["warnings"]) * 0.1
        
        validation_result["quality_score"] = max(0.0, min(1.0, base_quality + quality_adjustment))
        
        return validation_result

# Factory function
def create_text_formatter() -> TextFormatter:
    """
    Factory function to create a text formatter instance.
    
    Returns:
        TextFormatter instance
    """
    return TextFormatter()

if __name__ == "__main__":
    # Example usage
    formatter = create_text_formatter()
    
    # Sample processed text data
    sample_text = {
        "original_text": "This product is amazing! Great quality and fast shipping. Highly recommend!",
        "entities": [
            {"Text": "product", "Type": "PRODUCT", "Score": 0.95},
            {"Text": "shipping", "Type": "SERVICE", "Score": 0.88}
        ],
        "sentiment": {"Sentiment": "POSITIVE", "Score": 0.92},
        "key_phrases": [
            {"Text": "great quality", "Score": 0.94},
            {"Text": "fast shipping", "Score": 0.89}
        ],
        "metadata": {
            "customer_id": "CUST-00001",
            "product_id": "PROD-12345",
            "review_date": "2023-12-01",
            "quality_score": 0.91
        }
    }
    
    # Validate data
    validation = formatter.validate_text_data(sample_text)
    print("Validation result:")
    print(json.dumps(validation, indent=2))
    
    # Format for Claude
    claude_formatted = formatter.format_for_claude(sample_text, "conversation")
    print("\nClaude formatted (conversation):")
    print(json.dumps(claude_formatted, indent=2))
    
    # Format for training
    training_formatted = formatter.format_for_training(sample_text, "jsonl")
    print("\nTraining formatted (JSONL):")
    print(training_formatted["jsonl_line"])