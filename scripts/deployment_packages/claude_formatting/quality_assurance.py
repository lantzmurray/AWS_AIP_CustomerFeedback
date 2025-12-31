#!/usr/bin/env python3
"""
Quality Assurance Component

Validates formatted data against foundation model requirements with comprehensive
schema validation and quality scoring capabilities.
"""

import json
import boto3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import re
import jsonschema
from decimal import Decimal

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class QualityAssurance:
    """
    Provides quality assurance and validation for formatted data.
    """
    
    def __init__(self):
        """Initialize the quality assurance component."""
        self.s3_client = boto3.client('s3')
        self.cloudwatch_client = boto3.client('cloudwatch')
        self.schemas = self._load_schemas()
        self.validation_rules = self._load_validation_rules()
        
    def _load_schemas(self) -> Dict[str, Dict]:
        """Load JSON schemas for different model formats."""
        return {
            "claude_conversation": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "anthropic_version": {
                        "type": "string",
                        "enum": ["bedrock-2023-05-31"]
                    },
                    "max_tokens": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100000
                    },
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["user", "assistant"]
                                },
                                "content": {
                                    "type": ["string", "array"]
                                }
                            },
                            "required": ["role", "content"]
                        },
                        "minItems": 1
                    },
                    "temperature": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "top_p": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["anthropic_version", "max_tokens", "messages"]
            },
            "titan_text": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "inputText": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 8000
                    },
                    "textGenerationConfig": {
                        "type": "object",
                        "properties": {
                            "maxTokenCount": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 8000
                            },
                            "temperature": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0
                            },
                            "topP": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0
                            }
                        },
                        "required": ["maxTokenCount"]
                    }
                },
                "required": ["inputText", "textGenerationConfig"]
            },
            "training_jsonl": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "minLength": 1
                    },
                    "completion": {
                        "type": "string",
                        "minLength": 1
                    },
                    "data_type": {
                        "type": "string",
                        "enum": ["text", "image", "audio", "survey"]
                    },
                    "model": {
                        "type": "string",
                        "minLength": 1
                    },
                    "quality_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["prompt", "completion", "data_type", "model"]
            }
        }
    
    def _load_validation_rules(self) -> Dict[str, Dict]:
        """Load validation rules for different data types."""
        return {
            "text": {
                "min_length": 10,
                "max_length": 50000,
                "required_fields": ["original_text", "sentiment", "metadata"],
                "quality_thresholds": {
                    "min_quality_score": 0.6,
                    "min_sentiment_confidence": 0.7
                }
            },
            "image": {
                "min_labels": 1,
                "max_labels": 50,
                "required_fields": ["extracted_text", "labels", "metadata"],
                "quality_thresholds": {
                    "min_quality_score": 0.6,
                    "min_label_confidence": 0.7
                }
            },
            "audio": {
                "min_duration": 10,
                "max_duration": 3600,
                "required_fields": ["transcript", "sentiment", "metadata"],
                "quality_thresholds": {
                    "min_quality_score": 0.6,
                    "min_sentiment_confidence": 0.7
                }
            },
            "survey": {
                "min_ratings": 1,
                "max_ratings": 20,
                "required_fields": ["summary_text", "ratings", "metadata"],
                "quality_thresholds": {
                    "min_quality_score": 0.6,
                    "min_rating_value": 1.0,
                    "max_rating_value": 5.0
                }
            }
        }
    
    def validate_schema_compliance(self, formatted_data: Dict[str, Any], 
                               model_format: str) -> Dict[str, Any]:
        """
        Check schema adherence for formatted data.
        
        Args:
            formatted_data: Formatted data to validate
            model_format: Target model format (claude_conversation, titan_text, etc.)
            
        Returns:
            Schema validation result
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "schema_version": "draft-07"
        }
        
        # Get schema for model format
        schema = self.schemas.get(model_format)
        if not schema:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Unknown model format: {model_format}")
            return validation_result
        
        try:
            # Validate against schema
            jsonschema.validate(formatted_data, schema)
            validation_result["message"] = "Schema validation passed"
        except jsonschema.ValidationError as e:
            validation_result["valid"] = False
            validation_result["errors"].append({
                "path": "->".join(str(p) for p in e.path) if e.path else "root",
                "message": e.message,
                "schema_path": "->".join(str(p) for p in e.schema_path) if e.schema_path else "root",
                "failed_value": e.instance
            })
        except jsonschema.SchemaError as e:
            validation_result["valid"] = False
            validation_result["errors"].append({
                "type": "schema_error",
                "message": str(e)
            })
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append({
                "type": "validation_error",
                "message": f"Unexpected error: {str(e)}"
            })
        
        return validation_result
    
    def validate_token_limits(self, formatted_data: Dict[str, Any], 
                           model_type: str) -> Dict[str, Any]:
        """
        Ensure requests stay within model token limits.
        
        Args:
            formatted_data: Formatted data to validate
            model_type: Target model type
            
        Returns:
            Token limit validation result
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "token_count": 0,
            "limit": 0,
            "utilization": 0.0
        }
        
        # Define token limits for different models
        token_limits = {
            "claude-v2": 100000,
            "claude-instant-v1": 100000,
            "titan-text-express-v1": 8000,
            "jurassic-2-mid-v1": 8192
        }
        
        limit = token_limits.get(model_type, 8000)
        validation_result["limit"] = limit
        
        # Estimate token count
        token_count = self._estimate_token_count(formatted_data)
        validation_result["token_count"] = token_count
        
        # Calculate utilization
        utilization = token_count / limit if limit > 0 else 0
        validation_result["utilization"] = utilization
        
        # Check if within limits
        if token_count > limit:
            validation_result["valid"] = False
            validation_result["issues"].append(
                f"Token count ({token_count}) exceeds model limit ({limit})"
            )
        elif utilization > 0.95:
            validation_result["issues"].append(
                f"Token utilization ({utilization:.1%}) is very close to limit"
            )
        
        return validation_result
    
    def validate_content_quality(self, formatted_data: Dict[str, Any], 
                             data_type: str) -> Dict[str, Any]:
        """
        Check content appropriateness and quality.
        
        Args:
            formatted_data: Formatted data to validate
            data_type: Type of data (text, image, audio, survey)
            
        Returns:
            Content quality validation result
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "quality_score": 0.0,
            "content_analysis": {}
        }
        
        # Extract text content for analysis
        text_content = self._extract_text_content(formatted_data, data_type)
        
        if not text_content:
            validation_result["warnings"].append("No text content found for quality analysis")
            return validation_result
        
        # Perform content quality checks
        quality_checks = {
            "length_appropriate": self._check_length_appropriateness(text_content, data_type),
            "language_quality": self._check_language_quality(text_content),
            "content_completeness": self._check_content_completeness(formatted_data, data_type),
            "structure_quality": self._check_structure_quality(formatted_data, data_type)
        }
        
        validation_result["content_analysis"] = quality_checks
        
        # Calculate overall quality score
        quality_scores = [
            quality_checks["length_appropriate"]["score"],
            quality_checks["language_quality"]["score"],
            quality_checks["content_completeness"]["score"],
            quality_checks["structure_quality"]["score"]
        ]
        
        overall_score = sum(quality_scores) / len(quality_scores)
        validation_result["quality_score"] = overall_score
        
        # Check quality thresholds
        rules = self.validation_rules.get(data_type, {})
        thresholds = rules.get("quality_thresholds", {})
        min_quality = thresholds.get("min_quality_score", 0.6)
        
        if overall_score < min_quality:
            validation_result["valid"] = False
            validation_result["issues"].append(
                f"Content quality score ({overall_score:.2f}) below threshold ({min_quality})"
            )
        
        # Add specific issues from quality checks
        for check_name, check_result in quality_checks.items():
            if not check_result["passed"]:
                validation_result["issues"].extend(check_result["issues"])
            if check_result["warnings"]:
                validation_result["warnings"].extend(check_result["warnings"])
        
        return validation_result
    
    def validate_metadata_completeness(self, formatted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify metadata requirements are met.
        
        Args:
            formatted_data: Formatted data to validate
            
        Returns:
            Metadata completeness validation result
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "completeness_score": 0.0,
            "missing_fields": [],
            "present_fields": []
        }
        
        # Check for metadata section
        metadata = formatted_data.get("metadata", {})
        if not metadata:
            validation_result["valid"] = False
            validation_result["issues"].append("No metadata section found")
            return validation_result
        
        # Define required metadata fields
        required_metadata_fields = [
            "data_type",
            "model",
            "format_timestamp",
            "source_data_id"
        ]
        
        # Check required fields
        present_fields = []
        missing_fields = []
        
        for field in required_metadata_fields:
            if field in metadata and metadata[field]:
                present_fields.append(field)
            else:
                missing_fields.append(field)
        
        validation_result["present_fields"] = present_fields
        validation_result["missing_fields"] = missing_fields
        
        # Calculate completeness score
        total_fields = len(required_metadata_fields)
        completeness_score = len(present_fields) / total_fields if total_fields > 0 else 0
        validation_result["completeness_score"] = completeness_score
        
        # Add issues for missing fields
        if missing_fields:
            validation_result["valid"] = False
            validation_result["issues"].append(
                f"Missing required metadata fields: {', '.join(missing_fields)}"
            )
        
        # Check optional but recommended fields
        recommended_fields = [
            "quality_score",
            "customer_id",
            "product_id",
            "language",
            "region"
        ]
        
        missing_recommended = [
            field for field in recommended_fields 
            if field not in metadata or not metadata[field]
        ]
        
        if missing_recommended:
            validation_result["warnings"].append(
                f"Missing recommended metadata fields: {', '.join(missing_recommended)}"
            )
        
        return validation_result
    
    def generate_validation_report(self, formatted_data: Dict[str, Any], 
                              model_format: str, data_type: str, 
                              model_type: str) -> Dict[str, Any]:
        """
        Create comprehensive quality report.
        
        Args:
            formatted_data: Formatted data to validate
            model_format: Target model format
            data_type: Type of data
            model_type: Target model type
            
        Returns:
            Comprehensive validation report
        """
        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "data_id": formatted_data.get("metadata", {}).get("source_data_id", "unknown"),
            "model_format": model_format,
            "data_type": data_type,
            "model_type": model_type,
            "overall_valid": True,
            "overall_score": 0.0,
            "validation_results": {},
            "recommendations": [],
            "quality_metrics": {}
        }
        
        # Run all validations
        schema_result = self.validate_schema_compliance(formatted_data, model_format)
        token_result = self.validate_token_limits(formatted_data, model_type)
        content_result = self.validate_content_quality(formatted_data, data_type)
        metadata_result = self.validate_metadata_completeness(formatted_data)
        
        report["validation_results"] = {
            "schema_compliance": schema_result,
            "token_limits": token_result,
            "content_quality": content_result,
            "metadata_completeness": metadata_result
        }
        
        # Calculate overall validity
        overall_valid = all([
            schema_result["valid"],
            token_result["valid"],
            content_result["valid"],
            metadata_result["valid"]
        ])
        report["overall_valid"] = overall_valid
        
        # Calculate overall score
        scores = [
            1.0 if schema_result["valid"] else 0.0,
            1.0 if token_result["valid"] else 0.0,
            content_result.get("quality_score", 0.0),
            metadata_result["completeness_score"]
        ]
        
        overall_score = sum(scores) / len(scores)
        report["overall_score"] = overall_score
        
        # Generate recommendations
        recommendations = []
        
        if not schema_result["valid"]:
            recommendations.append("Fix schema compliance issues before processing")
        
        if not token_result["valid"]:
            recommendations.append("Reduce content length to meet token limits")
        elif token_result["utilization"] > 0.9:
            recommendations.append("Consider reducing content to improve token efficiency")
        
        if not content_result["valid"]:
            recommendations.append("Improve content quality and completeness")
        
        if not metadata_result["valid"]:
            recommendations.append("Add missing required metadata fields")
        
        report["recommendations"] = recommendations
        
        # Calculate quality metrics
        quality_metrics = {
            "schema_compliance": 1.0 if schema_result["valid"] else 0.0,
            "token_efficiency": 1.0 - min(token_result["utilization"], 1.0),
            "content_quality": content_result.get("quality_score", 0.0),
            "metadata_completeness": metadata_result["completeness_score"]
        }
        report["quality_metrics"] = quality_metrics
        
        return report
    
    def send_metrics_to_cloudwatch(self, validation_report: Dict[str, Any]):
        """
        Send validation metrics to CloudWatch.
        
        Args:
            validation_report: Validation report with metrics
        """
        try:
            metrics = []
            
            # Overall validation metrics
            metrics.append({
                "MetricName": "ValidationSuccess",
                "Value": 1 if validation_report["overall_valid"] else 0,
                "Unit": "Count",
                "Dimensions": [
                    {"Name": "ModelFormat", "Value": validation_report["model_format"]},
                    {"Name": "DataType", "Value": validation_report["data_type"]}
                ]
            })
            
            # Quality score metrics
            metrics.append({
                "MetricName": "ValidationScore",
                "Value": validation_report["overall_score"],
                "Unit": "None",
                "Dimensions": [
                    {"Name": "ModelFormat", "Value": validation_report["model_format"]},
                    {"Name": "DataType", "Value": validation_report["data_type"]}
                ]
            })
            
            # Individual quality metrics
            quality_metrics = validation_report.get("quality_metrics", {})
            for metric_name, value in quality_metrics.items():
                metrics.append({
                    "MetricName": f"Validation{metric_name.title()}",
                    "Value": value,
                    "Unit": "None",
                    "Dimensions": [
                        {"Name": "ModelFormat", "Value": validation_report["model_format"]},
                        {"Name": "DataType", "Value": validation_report["data_type"]}
                    ]
                })
            
            # Send metrics to CloudWatch
            self.cloudwatch_client.put_metric_data(
                Namespace="CustomerFeedback/Formatting/Validation",
                MetricData=metrics
            )
            
            logger.info(f"Sent {len(metrics)} validation metrics to CloudWatch")
            
        except Exception as e:
            logger.error(f"Error sending validation metrics to CloudWatch: {str(e)}")
    
    def create_quality_threshold_alerts(self, validation_report: Dict[str, Any]):
        """
        Create alerts for quality threshold violations.
        
        Args:
            validation_report: Validation report to check
        """
        alerts = []
        
        # Check for critical issues
        if not validation_report["overall_valid"]:
            alerts.append({
                "severity": "CRITICAL",
                "title": "Validation Failed",
                "message": f"Data validation failed for {validation_report['data_type']} data",
                "recommendation": "Review and fix validation issues before processing",
                "timestamp": datetime.now().isoformat()
            })
        
        # Check quality score threshold
        if validation_report["overall_score"] < 0.7:
            alerts.append({
                "severity": "WARNING",
                "title": "Low Quality Score",
                "message": f"Quality score ({validation_report['overall_score']:.2f}) below threshold",
                "recommendation": "Review content quality and completeness",
                "timestamp": datetime.now().isoformat()
            })
        
        # Check token utilization
        token_result = validation_report["validation_results"].get("token_limits", {})
        if token_result.get("utilization", 0) > 0.95:
            alerts.append({
                "severity": "WARNING",
                "title": "High Token Utilization",
                "message": f"Token utilization ({token_result['utilization']:.1%}) very high",
                "recommendation": "Consider reducing content length",
                "timestamp": datetime.now().isoformat()
            })
        
        # Store alerts if any
        if alerts:
            self._store_quality_alerts(alerts, validation_report)
    
    def _estimate_token_count(self, formatted_data: Dict[str, Any]) -> int:
        """Estimate token count for formatted data."""
        # Extract text content
        text_content = json.dumps(formatted_data)
        
        # Simple token estimation (rough approximation)
        # In reality, would use model-specific tokenizer
        words = text_content.split()
        tokens = 0
        
        for word in words:
            # Approximate tokens per word
            if len(word) <= 4:
                tokens += 1
            elif len(word) <= 8:
                tokens += 2
            else:
                tokens += 3
        
        # Add some buffer for special tokens and formatting
        tokens = int(tokens * 1.3)
        
        return tokens
    
    def _extract_text_content(self, formatted_data: Dict[str, Any], data_type: str) -> str:
        """Extract text content from formatted data."""
        if data_type == "text":
            # Look for text in various fields
            if "messages" in formatted_data:
                # Claude format
                messages = formatted_data["messages"]
                text_parts = []
                for message in messages:
                    content = message.get("content", "")
                    if isinstance(content, str):
                        text_parts.append(content)
                    elif isinstance(content, list):
                        # Multimodal content
                        for item in content:
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                return " ".join(text_parts)
            elif "inputText" in formatted_data:
                # Titan format
                return formatted_data["inputText"]
            elif "prompt" in formatted_data:
                # Training format
                return formatted_data["prompt"]
        
        elif data_type == "image":
            # Extract text from image data
            if "messages" in formatted_data:
                messages = formatted_data["messages"]
                for message in messages:
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                return item.get("text", "")
            elif "inputText" in formatted_data:
                return formatted_data["inputText"]
        
        elif data_type == "audio":
            # Extract transcript
            if "messages" in formatted_data:
                messages = formatted_data["messages"]
                for message in messages:
                    content = message.get("content", "")
                    if isinstance(content, str):
                        return content
            elif "inputText" in formatted_data:
                return formatted_data["inputText"]
        
        elif data_type == "survey":
            # Extract survey content
            if "messages" in formatted_data:
                messages = formatted_data["messages"]
                for message in messages:
                    content = message.get("content", "")
                    if isinstance(content, str):
                        return content
            elif "inputText" in formatted_data:
                return formatted_data["inputText"]
        
        return ""
    
    def _check_length_appropriateness(self, text: str, data_type: str) -> Dict[str, Any]:
        """Check if text length is appropriate for data type."""
        rules = self.validation_rules.get(data_type, {})
        min_length = rules.get("min_length", 10)
        max_length = rules.get("max_length", 50000)
        
        text_length = len(text)
        
        result = {
            "passed": True,
            "score": 1.0,
            "issues": [],
            "warnings": [],
            "actual_length": text_length,
            "min_length": min_length,
            "max_length": max_length
        }
        
        if text_length < min_length:
            result["passed"] = False
            result["score"] = 0.5
            result["issues"].append(f"Text too short: {text_length} < {min_length}")
        elif text_length > max_length:
            result["passed"] = False
            result["score"] = 0.5
            result["issues"].append(f"Text too long: {text_length} > {max_length}")
        elif text_length < min_length * 2:
            result["warnings"].append(f"Text relatively short: {text_length}")
        
        return result
    
    def _check_language_quality(self, text: str) -> Dict[str, Any]:
        """Check language quality indicators."""
        result = {
            "passed": True,
            "score": 1.0,
            "issues": [],
            "warnings": [],
            "metrics": {}
        }
        
        # Check for basic language quality issues
        issues = []
        warnings = []
        
        # Check for excessive repetition
        words = text.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_words = [word for word, count in word_counts.items() if count > 5]
        if repeated_words:
            warnings.append(f"Excessive repetition detected: {', '.join(repeated_words[:3])}")
        
        # Check for very short sentences
        sentences = re.split(r'[.!?]+', text)
        short_sentences = [s for s in sentences if len(s.strip()) < 5]
        if len(short_sentences) / len(sentences) > 0.5:
            warnings.append("Many very short sentences detected")
        
        # Check for special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:]', text)) / len(text)
        if special_char_ratio > 0.1:
            warnings.append("High ratio of special characters detected")
        
        # Calculate quality score
        score = 1.0
        if warnings:
            score -= len(warnings) * 0.1
        
        result["score"] = max(0.0, score)
        result["warnings"] = warnings
        result["metrics"] = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "repetition_detected": len(repeated_words) > 0,
            "special_char_ratio": special_char_ratio
        }
        
        if result["score"] < 0.7:
            result["passed"] = False
            result["issues"].extend(warnings)
        
        return result
    
    def _check_content_completeness(self, formatted_data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Check content completeness based on data type."""
        result = {
            "passed": True,
            "score": 1.0,
            "issues": [],
            "warnings": []
        }
        
        rules = self.validation_rules.get(data_type, {})
        required_fields = rules.get("required_fields", [])
        
        present_fields = []
        missing_fields = []
        
        for field in required_fields:
            if field in formatted_data and formatted_data[field]:
                present_fields.append(field)
            else:
                missing_fields.append(field)
        
        completeness_score = len(present_fields) / len(required_fields) if required_fields else 1.0
        result["score"] = completeness_score
        
        if missing_fields:
            result["passed"] = False
            result["issues"].append(f"Missing required fields: {', '.join(missing_fields)}")
        
        return result
    
    def _check_structure_quality(self, formatted_data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Check structural quality of formatted data."""
        result = {
            "passed": True,
            "score": 1.0,
            "issues": [],
            "warnings": []
        }
        
        # Check for proper JSON structure
        try:
            json_str = json.dumps(formatted_data)
            json.loads(json_str)
        except json.JSONDecodeError as e:
            result["passed"] = False
            result["score"] = 0.0
            result["issues"].append(f"Invalid JSON structure: {str(e)}")
            return result
        
        # Check for empty or null values in critical fields
        critical_fields = ["messages", "inputText", "prompt"]
        empty_critical = []
        
        for field in critical_fields:
            if field in formatted_data:
                value = formatted_data[field]
                if not value or (isinstance(value, str) and not value.strip()):
                    empty_critical.append(field)
        
        if empty_critical:
            result["passed"] = False
            result["score"] = 0.5
            result["issues"].append(f"Empty critical fields: {', '.join(empty_critical)}")
        
        return result
    
    def _store_quality_alerts(self, alerts: List[Dict[str, Any]], 
                            validation_report: Dict[str, Any]):
        """Store quality alerts for monitoring."""
        try:
            # Create alert record
            alert_record = {
                "timestamp": datetime.now().isoformat(),
                "data_id": validation_report.get("data_id", "unknown"),
                "model_format": validation_report.get("model_format", "unknown"),
                "data_type": validation_report.get("data_type", "unknown"),
                "alerts": alerts,
                "validation_summary": {
                    "overall_valid": validation_report.get("overall_valid", False),
                    "overall_score": validation_report.get("overall_score", 0.0)
                }
            }
            
            # Store in S3 for audit trail
            alert_key = f"quality-alerts/{validation_report.get('data_type', 'unknown')}/{datetime.now().strftime('%Y/%m/%d')}/alert_{validation_report.get('data_id', 'unknown')}_{int(datetime.now().timestamp())}.json"
            
            self.s3_client.put_object(
                Bucket=os.environ.get('ALERTS_BUCKET', 'customer-feedback-analysis-alerts'),
                Key=alert_key,
                Body=json.dumps(alert_record, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Stored {len(alerts)} quality alerts to S3")
            
        except Exception as e:
            logger.error(f"Error storing quality alerts: {str(e)}")

# Factory function
def create_quality_assurance() -> QualityAssurance:
    """
    Factory function to create a quality assurance instance.
    
    Returns:
        QualityAssurance instance
    """
    return QualityAssurance()

if __name__ == "__main__":
    # Example usage
    qa = create_quality_assurance()
    
    # Sample formatted data
    sample_data = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": "Analyze this customer review: This product is amazing!"
            }
        ],
        "temperature": 0.7,
        "metadata": {
            "data_type": "text",
            "model": "claude-v2",
            "format_timestamp": datetime.now().isoformat(),
            "source_data_id": "CUST-00001"
        }
    }
    
    # Validate schema compliance
    schema_result = qa.validate_schema_compliance(sample_data, "claude_conversation")
    print("Schema validation result:")
    print(json.dumps(schema_result, indent=2))
    
    # Validate token limits
    token_result = qa.validate_token_limits(sample_data, "claude-v2")
    print("\nToken limit validation result:")
    print(json.dumps(token_result, indent=2))
    
    # Generate comprehensive report
    report = qa.generate_validation_report(sample_data, "claude_conversation", "text", "claude-v2")
    print("\nComprehensive validation report:")
    print(json.dumps(report, indent=2))