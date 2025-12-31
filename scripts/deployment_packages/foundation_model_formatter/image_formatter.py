#!/usr/bin/env python3
"""
Image Data Formatter

Specialized formatter for image metadata that handles multimodal format for 
vision-language models. Includes image embeddings, descriptions, and integration 
with Phase 2 image processing output.
"""

import json
import base64
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import boto3
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import io
from PIL import Image
import numpy as np

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ImageFormatter:
    """
    Specialized formatter for image data with multimodal capabilities.
    """
    
    def __init__(self):
        """Initialize the image formatter."""
        self.s3_client = boto3.client('s3')
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load image formatting templates."""
        return {
            "claude_multimodal": """You are analyzing product images and extracted text to provide customer feedback insights.

EXTRACTED TEXT:
{extracted_text}

DETECTED LABELS:
{labels}

DETECTED TEXT ELEMENTS:
{detected_text}

IMAGE ANALYSIS:
{image_analysis}

METADATA:
- Product ID: {product_id}
- Customer ID: {customer_id}
- File Type: {file_type}
- Quality Score: {quality_score}

Please analyze both the image and extracted text to provide:
1. Description of what the image shows
2. Key feedback points from extracted text
3. Product quality assessment
4. Customer concerns or issues identified
5. Recommended actions

Format your response as structured JSON with the following keys:
image_description, key_feedback_points, quality_assessment, concerns, recommended_actions""",
            
            "titan_analysis": """Analyze this product image and extract business insights:

Extracted Text: {extracted_text}
Detected Labels: {labels}
Image Quality: {quality_score}

Provide analysis in JSON format with: description, feedback_summary, quality_rating, action_items.""",
            
            "training_prompt": """Product image analysis:
Extracted text: {extracted_text}
Labels: {labels}
Image metadata: {metadata}""",
            
            "training_completion": """{image_description}. Key insights: {insights}. Quality assessment: {quality_assessment}."""
        }
    
    def format_for_claude(self, processed_image: Dict[str, Any], 
                         format_type: str = "multimodal") -> Dict[str, Any]:
        """
        Format processed image data for Claude models.
        
        Args:
            processed_image: Processed image data from Phase 2
            format_type: Output format (multimodal, jsonl, parquet)
            
        Returns:
            Formatted data for Claude
        """
        if format_type == "multimodal":
            return self._format_claude_multimodal(processed_image)
        elif format_type == "jsonl":
            return self._format_jsonl(processed_image, "claude")
        elif format_type == "parquet":
            return self._format_parquet(processed_image, "claude")
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def format_for_titan(self, processed_image: Dict[str, Any],
                        format_type: str = "json") -> Dict[str, Any]:
        """
        Format processed image data for Titan models.
        
        Args:
            processed_image: Processed image data from Phase 2
            format_type: Output format (json, jsonl, parquet)
            
        Returns:
            Formatted data for Titan
        """
        if format_type == "json":
            return self._format_titan_json(processed_image)
        elif format_type == "jsonl":
            return self._format_jsonl(processed_image, "titan")
        elif format_type == "parquet":
            return self._format_parquet(processed_image, "titan")
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def format_for_training(self, processed_image: Dict[str, Any],
                           output_format: str = "jsonl") -> Dict[str, Any]:
        """
        Format processed image data for model training.
        
        Args:
            processed_image: Processed image data from Phase 2
            output_format: Output format (jsonl, parquet)
            
        Returns:
            Training data format
        """
        if output_format == "jsonl":
            return self._format_training_jsonl(processed_image)
        elif output_format == "parquet":
            return self._format_training_parquet(processed_image)
        else:
            raise ValueError(f"Unsupported training format: {output_format}")
    
    def encode_image_base64(self, image_source: Union[str, bytes], 
                          compression_quality: int = 85) -> str:
        """
        Encode image to base64 with optional compression.
        
        Args:
            image_source: Image source (S3 path, local path, or bytes)
            compression_quality: JPEG compression quality (0-100)
            
        Returns:
            Base64 encoded image string
        """
        try:
            if isinstance(image_source, str):
                if image_source.startswith("s3://"):
                    # Get from S3
                    bucket, key = image_source[5:].split("/", 1)
                    response = self.s3_client.get_object(Bucket=bucket, Key=key)
                    image_bytes = response['Body'].read()
                else:
                    # Local file
                    with open(image_source, 'rb') as f:
                        image_bytes = f.read()
            else:
                # Raw bytes
                image_bytes = image_source
            
            # Compress image if needed
            if compression_quality < 100:
                image_bytes = self._compress_image(image_bytes, compression_quality)
            
            return base64.b64encode(image_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            return ""
    
    def extract_visual_features(self, processed_image: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract visual features from processed image data.
        
        Args:
            processed_image: Processed image data from Phase 2
            
        Returns:
            Visual features dictionary
        """
        features = {
            "dominant_colors": [],
            "object_count": 0,
            "text_density": 0.0,
            "quality_indicators": {},
            "composition_analysis": {}
        }
        
        # Extract from labels
        labels = processed_image.get("labels", [])
        features["object_count"] = len(labels)
        
        # Extract from detected text
        detected_text = processed_image.get("detected_text", [])
        total_text_length = sum(len(text.get("DetectedText", "")) for text in detected_text)
        features["text_density"] = total_text_length / 1000.0  # Normalize
        
        # Quality indicators from metadata
        metadata = processed_image.get("metadata", {})
        features["quality_indicators"] = {
            "sharpness": metadata.get("sharpness_score", 0.0),
            "brightness": metadata.get("brightness_score", 0.0),
            "contrast": metadata.get("contrast_score", 0.0),
            "overall_quality": metadata.get("quality_score", 0.0)
        }
        
        return features
    
    def _format_claude_multimodal(self, processed_image: Dict[str, Any]) -> Dict[str, Any]:
        """Format image data for Claude multimodal analysis."""
        
        # Extract data
        extracted_text = processed_image.get("extracted_text", "")
        labels = processed_image.get("labels", [])
        detected_text = processed_image.get("detected_text", [])
        metadata = processed_image.get("metadata", {})
        
        # Format labels
        labels_text = json.dumps([label.get("Name", "") for label in labels], indent=2)
        
        # Format detected text
        detected_text_list = [text.get("DetectedText", "") for text in detected_text]
        detected_text_formatted = json.dumps(detected_text_list, indent=2)
        
        # Extract visual features
        visual_features = self.extract_visual_features(processed_image)
        image_analysis = json.dumps(visual_features, indent=2)
        
        # Extract metadata
        product_id = metadata.get("product_id", "N/A")
        customer_id = metadata.get("customer_id", "N/A")
        file_type = metadata.get("file_extension", "N/A")
        quality_score = metadata.get("quality_score", 0.0)
        
        # Create prompt
        prompt = self.templates["claude_multimodal"].format(
            extracted_text=extracted_text,
            labels=labels_text,
            detected_text=detected_text_formatted,
            image_analysis=image_analysis,
            product_id=product_id,
            customer_id=customer_id,
            file_type=file_type,
            quality_score=quality_score
        )
        
        # Create multimodal content
        content = [
            {"type": "text", "text": prompt}
        ]
        
        # Add image if available
        image_base64 = processed_image.get("image_base64")
        if not image_base64 and "s3_bucket" in metadata and "s3_key" in metadata:
            image_base64 = self.encode_image_base64(
                f"s3://{metadata['s3_bucket']}/{metadata['s3_key']}"
            )
        
        if image_base64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_base64
                }
            })
        
        # Create Claude multimodal request
        multimodal_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "messages": [
                {"role": "user", "content": content}
            ],
            "temperature": 0.7,
            "top_p": 0.999,
            "top_k": 250
        }
        
        # Add metadata
        multimodal_request["metadata"] = {
            "data_type": "image",
            "model": "claude-v2",
            "format_timestamp": datetime.now().isoformat(),
            "source_data_id": metadata.get("id", customer_id),
            "quality_score": quality_score,
            "visual_features": visual_features
        }
        
        return multimodal_request
    
    def _format_titan_json(self, processed_image: Dict[str, Any]) -> Dict[str, Any]:
        """Format image data for Titan Text model (text-only)."""
        
        extracted_text = processed_image.get("extracted_text", "")
        labels = processed_image.get("labels", [])
        metadata = processed_image.get("metadata", {})
        
        # Format for Titan
        labels_text = ", ".join([label.get("Name", "") for label in labels])
        quality_score = metadata.get("quality_score", 0.0)
        
        # Create Titan prompt
        prompt = self.templates["titan_analysis"].format(
            extracted_text=extracted_text,
            labels=labels_text,
            quality_score=quality_score
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
        titan_request["metadata"] = {
            "data_type": "image",
            "model": "titan-text-express-v1",
            "format_timestamp": datetime.now().isoformat(),
            "source_data_id": metadata.get("id", metadata.get("customer_id", "unknown")),
            "quality_score": quality_score
        }
        
        return titan_request
    
    def _format_jsonl(self, processed_image: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Format image data as JSONL for training."""
        
        extracted_text = processed_image.get("extracted_text", "")
        labels = processed_image.get("labels", [])
        metadata = processed_image.get("metadata", {})
        
        # Create training prompt
        labels_text = ", ".join([label.get("Name", "") for label in labels])
        metadata_text = json.dumps(metadata)
        
        prompt = self.templates["training_prompt"].format(
            extracted_text=extracted_text,
            labels=labels_text,
            metadata=metadata_text
        )
        
        # Generate completion
        completion = self.templates["training_completion"].format(
            image_description="Product image with visual elements",
            insights="Customer feedback from image analysis",
            quality_assessment=f"Quality score: {metadata.get('quality_score', 0.0)}"
        )
        
        jsonl_record = {
            "prompt": prompt,
            "completion": completion,
            "data_type": "image",
            "model": model,
            "quality_score": metadata.get("quality_score", 0.0),
            "customer_id": metadata.get("customer_id", ""),
            "product_id": metadata.get("product_id", ""),
            "file_type": metadata.get("file_extension", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        return {"jsonl_line": json.dumps(jsonl_record)}
    
    def _format_parquet(self, processed_image: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Format image data as Parquet for analytics."""
        
        extracted_text = processed_image.get("extracted_text", "")
        labels = processed_image.get("labels", [])
        detected_text = processed_image.get("detected_text", [])
        metadata = processed_image.get("metadata", {})
        
        # Extract visual features
        visual_features = self.extract_visual_features(processed_image)
        
        # Create DataFrame
        df_data = {
            "extracted_text": [extracted_text],
            "labels": [", ".join([label.get("Name", "") for label in labels])],
            "detected_text": [", ".join([text.get("DetectedText", "") for text in detected_text])],
            "object_count": [len(labels)],
            "text_density": [visual_features["text_density"]],
            "data_type": ["image"],
            "model": [model],
            "quality_score": [metadata.get("quality_score", 0.0)],
            "customer_id": [metadata.get("customer_id", "")],
            "product_id": [metadata.get("product_id", "")],
            "file_type": [metadata.get("file_extension", "")],
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
    
    def _format_training_jsonl(self, processed_image: Dict[str, Any]) -> Dict[str, Any]:
        """Format image data specifically for training."""
        
        extracted_text = processed_image.get("extracted_text", "")
        labels = processed_image.get("labels", [])
        metadata = processed_image.get("metadata", {})
        
        # Create fine-tuning format
        training_record = {
            "instruction": "Analyze this product image and provide customer feedback insights.",
            "input": f"""Extracted Text: {extracted_text}
Labels: {', '.join([label.get('Name', '') for label in labels])}
Quality Score: {metadata.get('quality_score', 0.0)}""",
            "output": f"""Visual Analysis: Product image with {len(labels)} detected objects
Text Content: {extracted_text}
Quality Assessment: {metadata.get('quality_score', 0.0)}""",
            "data_type": "image_analysis",
            "quality_score": metadata.get("quality_score", 0.0)
        }
        
        return {"jsonl_line": json.dumps(training_record)}
    
    def _format_training_parquet(self, processed_image: Dict[str, Any]) -> Dict[str, Any]:
        """Format image data as Parquet for training."""
        
        extracted_text = processed_image.get("extracted_text", "")
        labels = processed_image.get("labels", [])
        metadata = processed_image.get("metadata", {})
        
        # Create training DataFrame
        df_data = {
            "instruction": ["Analyze this product image and provide customer feedback insights."],
            "input": [f"""Extracted Text: {extracted_text}
Labels: {', '.join([label.get('Name', '') for label in labels])}
Quality Score: {metadata.get('quality_score', 0.0)}"""],
            "output": [f"""Visual Analysis: Product image with {len(labels)} detected objects
Text Content: {extracted_text}
Quality Assessment: {metadata.get('quality_score', 0.0)}"""],
            "data_type": ["image_analysis"],
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
    
    def _compress_image(self, image_bytes: bytes, quality: int = 85) -> bytes:
        """
        Compress image to reduce size while maintaining quality.
        
        Args:
            image_bytes: Original image bytes
            quality: JPEG compression quality (0-100)
            
        Returns:
            Compressed image bytes
        """
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Compress
            compressed_buffer = io.BytesIO()
            image.save(compressed_buffer, format='JPEG', quality=quality, optimize=True)
            
            return compressed_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error compressing image: {str(e)}")
            return image_bytes
    
    def create_image_embeddings(self, processed_image: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create image embeddings for similarity search and clustering.
        
        Args:
            processed_image: Processed image data
            
        Returns:
            Image embeddings dictionary
        """
        # This would typically use a vision model like CLIP or ResNet
        # For now, we'll create a placeholder that could be integrated
        
        labels = processed_image.get("labels", [])
        detected_text = processed_image.get("detected_text", [])
        metadata = processed_image.get("metadata", {})
        
        # Create feature vector (simplified)
        features = {
            "label_features": [label.get("Confidence", 0.0) for label in labels[:10]],
            "text_features": [len(text.get("DetectedText", "")) for text in detected_text[:5]],
            "quality_features": [
                metadata.get("quality_score", 0.0),
                metadata.get("sharpness_score", 0.0),
                metadata.get("brightness_score", 0.0)
            ],
            "metadata_features": [
                len(metadata.get("customer_id", "")),
                len(metadata.get("product_id", "")),
                1.0 if metadata.get("file_extension") == "jpg" else 0.0
            ]
        }
        
        # Flatten features into a single vector
        embedding = []
        for feature_group in features.values():
            embedding.extend(feature_group)
        
        # Pad or truncate to fixed size (e.g., 128 dimensions)
        target_size = 128
        if len(embedding) < target_size:
            embedding.extend([0.0] * (target_size - len(embedding)))
        else:
            embedding = embedding[:target_size]
        
        return {
            "embedding": embedding,
            "embedding_size": len(embedding),
            "feature_groups": features
        }
    
    def validate_image_data(self, processed_image: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate processed image data for formatting requirements.
        
        Args:
            processed_image: Processed image data to validate
            
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
        required_fields = ["extracted_text", "labels", "metadata"]
        for field in required_fields:
            if field not in processed_image:
                validation_result["valid"] = False
                validation_result["issues"].append(f"Missing required field: {field}")
        
        # Check labels
        labels = processed_image.get("labels", [])
        if not labels:
            validation_result["warnings"].append("No labels detected in image")
        
        # Check text extraction
        extracted_text = processed_image.get("extracted_text", "")
        if not extracted_text:
            validation_result["warnings"].append("No text extracted from image")
        
        # Check image quality
        metadata = processed_image.get("metadata", {})
        quality_score = metadata.get("quality_score", 0.0)
        if quality_score < 0.5:
            validation_result["warnings"].append("Low image quality detected")
        
        # Calculate overall quality score
        base_quality = quality_score
        
        # Adjust quality based on validation results
        quality_adjustment = 0.0
        if validation_result["issues"]:
            quality_adjustment -= len(validation_result["issues"]) * 0.2
        if validation_result["warnings"]:
            quality_adjustment -= len(validation_result["warnings"]) * 0.1
        
        validation_result["quality_score"] = max(0.0, min(1.0, base_quality + quality_adjustment))
        
        return validation_result

# Factory function
def create_image_formatter() -> ImageFormatter:
    """
    Factory function to create an image formatter instance.
    
    Returns:
        ImageFormatter instance
    """
    return ImageFormatter()

if __name__ == "__main__":
    # Example usage
    formatter = create_image_formatter()
    
    # Sample processed image data
    sample_image = {
        "extracted_text": "Premium Quality Product - Made in USA",
        "labels": [
            {"Name": "Product", "Confidence": 0.95},
            {"Name": "Text", "Confidence": 0.88},
            {"Name": "Packaging", "Confidence": 0.82}
        ],
        "detected_text": [
            {"DetectedText": "Premium Quality", "Type": "LINE"},
            {"DetectedText": "Made in USA", "Type": "LINE"}
        ],
        "metadata": {
            "customer_id": "CUST-00002",
            "product_id": "PROD-67890",
            "file_extension": "jpg",
            "quality_score": 0.89,
            "sharpness_score": 0.92,
            "brightness_score": 0.85
        }
    }
    
    # Validate data
    validation = formatter.validate_image_data(sample_image)
    print("Validation result:")
    print(json.dumps(validation, indent=2))
    
    # Extract visual features
    features = formatter.extract_visual_features(sample_image)
    print("\nVisual features:")
    print(json.dumps(features, indent=2))
    
    # Format for Claude
    claude_formatted = formatter.format_for_claude(sample_image, "multimodal")
    print("\nClaude formatted (multimodal):")
    print(json.dumps(claude_formatted, indent=2))