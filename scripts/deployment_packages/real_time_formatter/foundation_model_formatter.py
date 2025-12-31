#!/usr/bin/env python3
"""
Foundation Model Formatter

Core formatter that handles formatting for different foundation models including
Claude, Titan Text, and multi-model formats. Supports JSONL, Parquet, and
Conversation format outputs.
"""

import json
import base64
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import os
import boto3
from typing import Dict, List, Any, Optional, Union
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class FoundationModelFormatter:
    """
    Main formatter class for foundation models with support for multiple output formats.
    """
    
    def __init__(self):
        """Initialize the formatter with model configurations."""
        self.model_configs = self._load_model_configs()
        self.s3_client = boto3.client('s3')
        
    def _load_model_configs(self) -> Dict[str, Dict]:
        """Load model configurations."""
        return {
            "claude-v2": {
                "max_tokens": 100000,
                "supports_multimodal": True,
                "input_format": "conversation",
                "cost_per_1k_tokens": 0.008,
                "temperature_range": (0.0, 1.0)
            },
            "claude-instant-v1": {
                "max_tokens": 100000,
                "supports_multimodal": False,
                "input_format": "conversation",
                "cost_per_1k_tokens": 0.0008,
                "temperature_range": (0.0, 1.0)
            },
            "titan-text-express-v1": {
                "max_tokens": 8000,
                "supports_multimodal": False,
                "input_format": "json",
                "cost_per_1k_tokens": 0.0008,
                "temperature_range": (0.0, 1.0)
            },
            "jurassic-2-mid-v1": {
                "max_tokens": 8192,
                "supports_multimodal": False,
                "input_format": "json",
                "cost_per_1k_tokens": 0.0125,
                "temperature_range": (0.0, 1.0)
            }
        }
    
    def format_for_claude(self, data: Dict[str, Any], data_type: str, 
                        format_type: str = "conversation") -> Dict[str, Any]:
        """
        Format data for Claude models.
        
        Args:
            data: Input data to format
            data_type: Type of data (text, image, audio, survey)
            format_type: Output format type (conversation, jsonl, parquet)
            
        Returns:
            Formatted data for Claude
        """
        model_config = self.model_configs["claude-v2"]
        
        if format_type == "conversation":
            return self._format_claude_conversation(data, data_type, model_config)
        elif format_type == "jsonl":
            return self._format_jsonl(data, data_type, "claude-v2")
        elif format_type == "parquet":
            return self._format_parquet(data, data_type, "claude-v2")
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def format_for_titan(self, data: Dict[str, Any], data_type: str,
                        format_type: str = "json") -> Dict[str, Any]:
        """
        Format data for Titan Text models.
        
        Args:
            data: Input data to format
            data_type: Type of data (text, image, audio, survey)
            format_type: Output format type (json, jsonl, parquet)
            
        Returns:
            Formatted data for Titan
        """
        model_config = self.model_configs["titan-text-express-v1"]
        
        if format_type == "json":
            return self._format_titan_json(data, data_type, model_config)
        elif format_type == "jsonl":
            return self._format_jsonl(data, data_type, "titan-text-express-v1")
        elif format_type == "parquet":
            return self._format_parquet(data, data_type, "titan-text-express-v1")
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def format_multimodal(self, data: Dict[str, Any], data_type: str,
                         target_model: str = "claude-v2") -> Dict[str, Any]:
        """
        Format data for multimodal models.
        
        Args:
            data: Input data to format
            data_type: Type of data (text, image, audio, survey)
            target_model: Target model for formatting
            
        Returns:
            Multimodal formatted data
        """
        if not self.model_configs.get(target_model, {}).get("supports_multimodal", False):
            logger.warning(f"Model {target_model} does not support multimodal input")
            # Fallback to text-only formatting
            return self.format_for_claude(data, data_type, "conversation")
        
        if data_type == "image":
            return self._format_multimodal_image(data)
        elif data_type == "audio":
            return self._format_multimodal_audio(data)
        elif data_type == "text":
            return self._format_multimodal_text(data)
        elif data_type == "survey":
            return self._format_multimodal_survey(data)
        else:
            raise ValueError(f"Unsupported data type for multimodal: {data_type}")
    
    def _format_claude_conversation(self, data: Dict[str, Any], data_type: str,
                                  model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Format data as Claude conversation."""
        
        system_message = self._get_system_message(data_type)
        user_message = self._get_user_message(data, data_type)
        
        conversation = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": model_config["max_tokens"],
            "messages": [
                {"role": "user", "content": f"{system_message}\n\n{user_message}"}
            ],
            "temperature": 0.7,
            "top_p": 0.999,
            "top_k": 250
        }
        
        # Add metadata
        conversation["metadata"] = {
            "data_type": data_type,
            "model": "claude-v2",
            "format_timestamp": datetime.now().isoformat(),
            "source_data_id": data.get("metadata", {}).get("id", "unknown")
        }
        
        return conversation
    
    def _format_titan_json(self, data: Dict[str, Any], data_type: str,
                          model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for Titan Text model."""
        
        input_text = self._extract_text_content(data, data_type)
        
        titan_request = {
            "inputText": input_text,
            "textGenerationConfig": {
                "maxTokenCount": model_config["max_tokens"],
                "temperature": 0.7,
                "topP": 0.999,
                "stopSequences": []
            }
        }
        
        # Add metadata
        titan_request["metadata"] = {
            "data_type": data_type,
            "model": "titan-text-express-v1",
            "format_timestamp": datetime.now().isoformat(),
            "source_data_id": data.get("metadata", {}).get("id", "unknown")
        }
        
        return titan_request
    
    def _format_jsonl(self, data: Dict[str, Any], data_type: str, model: str) -> Dict[str, Any]:
        """Format data as JSONL for training."""
        
        prompt = self._extract_text_content(data, data_type)
        completion = self._generate_completion(data, data_type)
        
        jsonl_record = {
            "prompt": prompt,
            "completion": completion,
            "data_type": data_type,
            "model": model,
            "quality_score": data.get("metadata", {}).get("quality_score", 0.0),
            "timestamp": datetime.now().isoformat()
        }
        
        return {"jsonl_line": json.dumps(jsonl_record)}
    
    def _format_parquet(self, data: Dict[str, Any], data_type: str, model: str) -> Dict[str, Any]:
        """Format data as Parquet for analytics."""
        
        # Create a pandas DataFrame with the data
        df_data = {
            "prompt": [self._extract_text_content(data, data_type)],
            "completion": [self._generate_completion(data, data_type)],
            "data_type": [data_type],
            "model": [model],
            "quality_score": [data.get("metadata", {}).get("quality_score", 0.0)],
            "customer_id": [data.get("metadata", {}).get("customer_id", "")],
            "product_id": [data.get("metadata", {}).get("product_id", "")],
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
    
    def _format_multimodal_image(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format image data for multimodal models."""
        
        # Get image base64 if not already present
        image_base64 = data.get("image_base64")
        if not image_base64 and "s3_bucket" in data and "s3_key" in data:
            image_base64 = self._get_image_from_s3(data["s3_bucket"], data["s3_key"])
        
        extracted_text = data.get("extracted_text", "")
        labels = data.get("labels", [])
        detected_text = data.get("detected_text", [])
        
        # Create multimodal content
        content = [
            {
                "type": "text",
                "text": f"""Analyze this product image and provide customer feedback insights.

EXTRACTED TEXT:
{extracted_text}

DETECTED LABELS:
{', '.join([label.get('Name', '') for label in labels])}

DETECTED TEXT ELEMENTS:
{', '.join([text.get('DetectedText', '') for text in detected_text])}

Please provide:
1. Description of what the image shows
2. Key feedback points from extracted text
3. Product quality assessment
4. Customer concerns or issues identified
5. Recommended actions

Format your response as structured JSON."""
            }
        ]
        
        if image_base64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_base64
                }
            })
        
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "messages": [
                {"role": "user", "content": content}
            ],
            "temperature": 0.7
        }
    
    def _format_multimodal_audio(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format audio data for multimodal models."""
        
        transcript = data.get("transcript", "")
        speakers = data.get("speakers", [])
        sentiment = data.get("sentiment", {})
        key_phrases = data.get("key_phrases", [])
        
        prompt = f"""Analyze this customer service call transcript for business insights.

CALL TRANSCRIPT:
{transcript}

SPEAKER INFORMATION:
{json.dumps(speakers, indent=2)}

ANALYSIS DATA:
- Overall Sentiment: {sentiment.get('Sentiment', 'Unknown')}
- Key Phrases: {', '.join([phrase.get('Text', '') for phrase in key_phrases])}

Please provide:
1. Summary of the call conversation
2. Customer satisfaction assessment
3. Key issues or concerns raised
4. Service quality evaluation
5. Improvement recommendations
6. Follow-up actions needed

Format your response as structured JSON."""

        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 3000,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
    
    def _format_multimodal_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format text data for multimodal models."""
        
        original_text = data.get("original_text", "")
        entities = data.get("entities", [])
        sentiment = data.get("sentiment", {})
        key_phrases = data.get("key_phrases", [])
        
        prompt = f"""Analyze this customer review and provide actionable business insights.

CUSTOMER REVIEW:
{original_text}

ANALYSIS DATA:
- Entities: {json.dumps(entities, indent=2)}
- Sentiment: {sentiment.get('Sentiment', 'Unknown')} (confidence: {sentiment.get('Score', 0)})
- Key Phrases: {', '.join([phrase.get('Text', '') for phrase in key_phrases])}

Please provide:
1. A summary of the customer's main points
2. Key insights for business improvement
3. Recommended actions
4. Sentiment analysis explanation
5. Priority level for follow-up (High/Medium/Low)

Format your response as structured JSON."""

        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
    
    def _format_multimodal_survey(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format survey data for multimodal models."""
        
        summary_text = data.get("summary_text", "")
        ratings = data.get("ratings", {})
        comments = data.get("comments", [])
        improvement_areas = data.get("improvement_areas", [])
        
        prompt = f"""Analyze this customer survey response and generate business insights.

SURVEY SUMMARY:
{summary_text}

RATINGS:
{json.dumps(ratings, indent=2)}

COMMENTS:
{', '.join(comments)}

IMPROVEMENT AREAS:
{', '.join(improvement_areas)}

Please provide:
1. Overall customer satisfaction assessment
2. Key themes and patterns
3. Specific improvement areas to address
4. Priority level for follow-up
5. Recommended action plan

Format your response as structured JSON."""

        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
    
    def _get_system_message(self, data_type: str) -> str:
        """Get system message based on data type."""
        system_messages = {
            "text": "You are an expert at analyzing customer feedback and providing actionable business insights.",
            "image": "You are an expert at analyzing product images and customer feedback to provide business insights.",
            "audio": "You are an expert at analyzing customer service conversations and providing business insights.",
            "survey": "You are an expert at analyzing customer survey responses and providing business insights."
        }
        return system_messages.get(data_type, "You are an expert at analyzing customer feedback.")
    
    def _get_user_message(self, data: Dict[str, Any], data_type: str) -> str:
        """Get user message based on data type and content."""
        if data_type == "text":
            return f"Analyze this customer review: {data.get('original_text', '')}"
        elif data_type == "image":
            return f"Analyze this product image: {data.get('extracted_text', '')}"
        elif data_type == "audio":
            return f"Analyze this call transcript: {data.get('transcript', '')}"
        elif data_type == "survey":
            return f"Analyze this survey response: {data.get('summary_text', '')}"
        else:
            return "Analyze this customer feedback data."
    
    def _extract_text_content(self, data: Dict[str, Any], data_type: str) -> str:
        """Extract text content from data based on type."""
        if data_type == "text":
            return data.get("original_text", "")
        elif data_type == "image":
            return data.get("extracted_text", "")
        elif data_type == "audio":
            return data.get("transcript", "")
        elif data_type == "survey":
            return data.get("summary_text", "")
        else:
            return str(data)
    
    def _generate_completion(self, data: Dict[str, Any], data_type: str) -> str:
        """Generate completion text for training data."""
        if data_type == "text":
            sentiment = data.get("sentiment", {}).get("Sentiment", "Unknown")
            return f"Customer sentiment: {sentiment}. Key feedback points identified."
        elif data_type == "image":
            labels = [label.get("Name", "") for label in data.get("labels", [])]
            return f"Image analysis: {', '.join(labels)}. Product feedback extracted."
        elif data_type == "audio":
            sentiment = data.get("sentiment", {}).get("Sentiment", "Unknown")
            return f"Call analysis: {sentiment} sentiment. Service insights generated."
        elif data_type == "survey":
            ratings = data.get("ratings", {})
            overall = ratings.get("overall_satisfaction", 0)
            return f"Survey analysis: {overall}/5 satisfaction. Key themes identified."
        else:
            return "Customer feedback analyzed and insights generated."
    
    def _get_image_from_s3(self, bucket: str, key: str) -> str:
        """Retrieve image from S3 and convert to base64."""
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            image_bytes = response['Body'].read()
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Error retrieving image from S3: {str(e)}")
            return ""
    
    def validate_format(self, formatted_data: Dict[str, Any], model: str) -> Dict[str, Any]:
        """
        Validate formatted data against model requirements.
        
        Args:
            formatted_data: Formatted data to validate
            model: Target model name
            
        Returns:
            Validation result with status and issues
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "token_count": 0
        }
        
        model_config = self.model_configs.get(model)
        if not model_config:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Unknown model: {model}")
            return validation_result
        
        # Check token count (simplified estimation)
        text_content = json.dumps(formatted_data)
        estimated_tokens = len(text_content.split()) * 1.3  # Rough estimation
        
        if estimated_tokens > model_config["max_tokens"]:
            validation_result["valid"] = False
            validation_result["issues"].append(
                f"Token count ({estimated_tokens}) exceeds model limit ({model_config['max_tokens']})"
            )
        
        validation_result["token_count"] = estimated_tokens
        
        # Check required fields based on model
        if model.startswith("claude"):
            if "messages" not in formatted_data:
                validation_result["valid"] = False
                validation_result["issues"].append("Claude format requires 'messages' field")
        elif model.startswith("titan"):
            if "inputText" not in formatted_data:
                validation_result["valid"] = False
                validation_result["issues"].append("Titan format requires 'inputText' field")
        
        return validation_result

# Factory function for creating formatter instances
def create_formatter(model_type: str = "claude-v2") -> FoundationModelFormatter:
    """
    Factory function to create a formatter instance.
    
    Args:
        model_type: Default model type for formatting
        
    Returns:
        FoundationModelFormatter instance
    """
    return FoundationModelFormatter()

if __name__ == "__main__":
    # Example usage
    formatter = create_formatter()
    
    # Sample text data
    sample_text_data = {
        "original_text": "This product is amazing! Great quality and fast shipping.",
        "entities": [{"Text": "product", "Type": "PRODUCT", "Score": 0.95}],
        "sentiment": {"Sentiment": "POSITIVE", "Score": 0.87},
        "key_phrases": [{"Text": "great quality", "Score": 0.92}],
        "metadata": {
            "customer_id": "CUST-00001",
            "product_id": "PROD-12345",
            "quality_score": 0.92
        }
    }
    
    # Format for Claude
    claude_formatted = formatter.format_for_claude(sample_text_data, "text")
    print("Claude formatted:")
    print(json.dumps(claude_formatted, indent=2))
    
    # Validate format
    validation = formatter.validate_format(claude_formatted, "claude-v2")
    print("\nValidation result:")
    print(json.dumps(validation, indent=2))