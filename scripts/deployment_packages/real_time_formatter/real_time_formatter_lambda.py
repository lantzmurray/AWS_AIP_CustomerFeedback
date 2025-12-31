#!/usr/bin/env python3
"""
Real-time Formatter Lambda Function

Provides low-latency formatting for real-time inference with support for
on-demand formatting, caching, and connection pooling for Bedrock API.
"""

import json
import boto3
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
bedrock_runtime = boto3.client('bedrock-runtime')
cloudwatch_client = boto3.client('cloudwatch')

# Cache for formatted requests (simple in-memory cache)
format_cache = {}
cache_ttl = 300  # 5 minutes

# Connection pool for Bedrock
bedrock_pool = ThreadPoolExecutor(max_workers=10)

# Import formatters
def get_formatter(data_type: str, model: str):
    """Get appropriate formatter for data type and model."""
    if data_type == "text":
        from text_formatter import create_text_formatter
        return create_text_formatter()
    elif data_type == "image":
        from image_formatter import create_image_formatter
        return create_image_formatter()
    elif data_type == "audio":
        from audio_formatter import create_audio_formatter
        return create_audio_formatter()
    elif data_type == "survey":
        from survey_formatter import create_survey_formatter
        return create_survey_formatter()
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

def get_model_selection():
    """Get model selection strategy."""
    from model_selection_strategy import ModelSelectionStrategy
    return ModelSelectionStrategy()

def lambda_handler(event, context):
    """
    Lambda handler for real-time formatting.
    
    Args:
        event: Lambda event (API Gateway or S3)
        context: Lambda context
        
    Returns:
        Formatted response
    """
    start_time = time.time()
    
    try:
        # Parse request
        if event.get("httpMethod") == "POST":
            # API Gateway request
            return handle_api_request(event, context)
        elif "Records" in event:
            # S3 event
            return handle_s3_event(event, context)
        else:
            # Direct invocation
            return handle_direct_invocation(event, context)
            
    except Exception as e:
        logger.error(f"Error in real-time formatter: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "error": "Internal server error",
                "message": str(e)
            })
        }
    finally:
        # Log processing time
        processing_time = (time.time() - start_time) * 1000
        send_timing_metrics(processing_time)

def handle_api_request(event, context) -> Dict[str, Any]:
    """
    Handle API Gateway request for real-time formatting.
    
    Args:
        event: API Gateway event
        context: Lambda context
        
    Returns:
        API response
    """
    # Parse request body
    try:
        body = json.loads(event.get("body", "{}"))
    except json.JSONDecodeError:
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "error": "Invalid JSON in request body"
            })
        }
    
    # Validate request
    validation_result = validate_api_request(body)
    if not validation_result["valid"]:
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "error": "Validation failed",
                "details": validation_result["errors"]
            })
        }
    
    # Extract request parameters
    data_type = body.get("data_type")
    data = body.get("data")
    model_preference = body.get("model", "auto")
    format_type = body.get("format_type", "conversation")
    use_cache = body.get("use_cache", True)
    
    # Check cache first
    cache_key = generate_cache_key(data_type, data, model_preference, format_type)
    if use_cache:
        cached_result = get_from_cache(cache_key)
        if cached_result:
            logger.info(f"Cache hit for key: {cache_key}")
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({
                    "result": cached_result,
                    "cached": True,
                    "processing_time_ms": 0
                })
            }
    
    # Process formatting request
    result = process_formatting_request(
        data_type, data, model_preference, format_type
    )
    
    # Cache result
    if use_cache and result.get("success"):
        put_in_cache(cache_key, result)
    
    # Send metrics
    send_formatting_metrics(data_type, model_preference, result)
    
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "result": result,
            "cached": False,
            "processing_time_ms": result.get("processing_time_ms", 0)
        })
    }

def handle_s3_event(event, context) -> Dict[str, Any]:
    """
    Handle S3 event for real-time formatting.
    
    Args:
        event: S3 event
        context: Lambda context
        
    Returns:
        Processing result
    """
    # Get S3 object information
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    key = record["s3"]["object"]["key"]
    
    # Only process specific prefixes for real-time
    if not key.startswith("real-time/"):
        logger.info(f"Skipping non-real-time object: {key}")
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Object not in real-time path"})
        }
    
    try:
        # Get object from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        
        # Determine data type from key
        data_type = determine_data_type_from_key(key)
        
        # Get model preference (default to auto)
        model_preference = data.get("model_preference", "auto")
        format_type = data.get("format_type", "conversation")
        
        # Process formatting
        result = process_formatting_request(
            data_type, data, model_preference, format_type
        )
        
        # Store formatted result
        if result.get("success"):
            output_key = key.replace("real-time/", "formatted/")
            store_formatted_result(result, bucket, output_key)
        
        # Send metrics
        send_formatting_metrics(data_type, model_preference, result)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Real-time formatting completed",
                "result": result
            })
        }
        
    except Exception as e:
        logger.error(f"Error processing S3 object {key}: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "Processing failed",
                "message": str(e)
            })
        }

def handle_direct_invocation(event, context) -> Dict[str, Any]:
    """
    Handle direct Lambda invocation.
    
    Args:
        event: Direct invocation event
        context: Lambda context
        
    Returns:
        Processing result
    """
    # Extract parameters
    data_type = event.get("data_type")
    data = event.get("data")
    model_preference = event.get("model", "auto")
    format_type = event.get("format_type", "conversation")
    
    # Process formatting
    result = process_formatting_request(
        data_type, data, model_preference, format_type
    )
    
    # Send metrics
    send_formatting_metrics(data_type, model_preference, result)
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "result": result
        })
    }

def process_formatting_request(data_type: str, data: Dict[str, Any], 
                           model_preference: str, format_type: str) -> Dict[str, Any]:
    """
    Process formatting request with optimal model selection.
    
    Args:
        data_type: Type of data to format
        data: Data to format
        model_preference: Preferred model or "auto"
        format_type: Output format type
        
    Returns:
        Formatting result
    """
    start_time = time.time()
    
    try:
        # Select optimal model
        model_selector = get_model_selection()
        selected_model = model_selector.select_model(data_type, model_preference)
        
        # Get appropriate formatter
        formatter = get_formatter(data_type, selected_model)
        
        # Format data
        if selected_model.startswith("claude"):
            formatted_data = formatter.format_for_claude(data, format_type)
        elif selected_model.startswith("titan"):
            formatted_data = formatter.format_for_titan(data, format_type)
        else:
            formatted_data = formatter.format_for_training(data, format_type)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "formatted_data": formatted_data,
            "selected_model": selected_model,
            "data_type": data_type,
            "format_type": format_type,
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in formatting request: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "data_type": data_type,
            "format_type": format_type,
            "processing_time_ms": int((time.time() - start_time) * 1000),
            "timestamp": datetime.now().isoformat()
        }

def send_to_foundation_model(formatted_data: Dict[str, Any], 
                           model: str) -> Dict[str, Any]:
    """
    Send formatted data to foundation model.
    
    Args:
        formatted_data: Formatted data for model
        model: Target model ID
        
    Returns:
        Model response
    """
    start_time = time.time()
    
    try:
        # Prepare request based on model
        if model.startswith("claude"):
            request_body = {
                "prompt": formatted_data.get("prompt", ""),
                "max_tokens_to_sample": formatted_data.get("max_tokens", 2000),
                "temperature": formatted_data.get("temperature", 0.7),
                "top_k": 250,
                "top_p": 0.999,
                "stop_sequences": ["\n\nHuman:"]
            }
        elif model.startswith("titan"):
            request_body = formatted_data
        else:
            # Default format
            request_body = formatted_data
        
        # Invoke model
        response = bedrock_runtime.invoke_model(
            body=json.dumps(request_body),
            modelId=model,
            accept="application/json",
            contentType="application/json"
        )
        
        # Parse response
        response_body = json.loads(response['body'].read().decode('utf-8'))
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "response": response_body,
            "model": model,
            "processing_time_ms": processing_time_ms,
            "usage": response_body.get("usage", {}),
            "request_id": response.get("ResponseMetadata", {}).get("RequestId", "")
        }
        
    except Exception as e:
        logger.error(f"Error invoking model {model}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "model": model,
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }

def validate_api_request(request_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate API request parameters.
    
    Args:
        request_body: Request body to validate
        
    Returns:
        Validation result
    """
    result = {"valid": True, "errors": []}
    
    # Check required fields
    required_fields = ["data_type", "data"]
    for field in required_fields:
        if field not in request_body:
            result["valid"] = False
            result["errors"].append(f"Missing required field: {field}")
    
    # Validate data_type
    valid_data_types = ["text", "image", "audio", "survey"]
    data_type = request_body.get("data_type")
    if data_type not in valid_data_types:
        result["valid"] = False
        result["errors"].append(f"Invalid data_type: {data_type}")
    
    # Validate format_type
    format_type = request_body.get("format_type", "conversation")
    valid_formats = ["conversation", "json", "jsonl", "parquet"]
    if format_type not in valid_formats:
        result["valid"] = False
        result["errors"].append(f"Invalid format_type: {format_type}")
    
    # Validate model preference
    model_preference = request_body.get("model", "auto")
    valid_models = ["auto", "claude-v2", "claude-instant-v1", "titan-text-express-v1"]
    if model_preference not in valid_models:
        result["valid"] = False
        result["errors"].append(f"Invalid model preference: {model_preference}")
    
    return result

def generate_cache_key(data_type: str, data: Dict[str, Any], 
                    model: str, format_type: str) -> str:
    """
    Generate cache key for formatted data.
    
    Args:
        data_type: Type of data
        data: Data content
        model: Target model
        format_type: Output format
        
    Returns:
        Cache key
    """
    import hashlib
    
    # Create a hash of the key parameters
    key_data = f"{data_type}:{model}:{format_type}:{str(sorted(data.items()))}"
    return hashlib.md5(key_data.encode()).hexdigest()

def get_from_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    Get formatted data from cache.
    
    Args:
        cache_key: Cache key
        
    Returns:
        Cached data or None if not found/expired
    """
    global format_cache
    
    if cache_key in format_cache:
        cache_entry = format_cache[cache_key]
        
        # Check if expired
        if time.time() - cache_entry["timestamp"] < cache_ttl:
            return cache_entry["data"]
        else:
            # Remove expired entry
            del format_cache[cache_key]
    
    return None

def put_in_cache(cache_key: str, data: Dict[str, Any]) -> None:
    """
    Store formatted data in cache.
    
    Args:
        cache_key: Cache key
        data: Data to cache
    """
    global format_cache
    
    format_cache[cache_key] = {
        "data": data,
        "timestamp": time.time()
    }
    
    # Limit cache size
    if len(format_cache) > 1000:
        # Remove oldest entries
        oldest_keys = sorted(
            format_cache.keys(),
            key=lambda k: format_cache[k]["timestamp"]
        )[:100]
        
        for key in oldest_keys:
            del format_cache[key]

def determine_data_type_from_key(key: str) -> str:
    """
    Determine data type from S3 object key.
    
    Args:
        key: S3 object key
        
    Returns:
        Data type
    """
    key_lower = key.lower()
    
    if "text" in key_lower:
        return "text"
    elif "image" in key_lower:
        return "image"
    elif "audio" in key_lower:
        return "audio"
    elif "survey" in key_lower:
        return "survey"
    else:
        return "unknown"

def store_formatted_result(result: Dict[str, Any], bucket: str, key: str) -> None:
    """
    Store formatted result in S3.
    
    Args:
        result: Formatting result to store
        bucket: S3 bucket name
        key: S3 object key
    """
    try:
        # Store the formatted data
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(result, indent=2),
            ContentType='application/json'
        )
        
        logger.info(f"Stored formatted result: s3://{bucket}/{key}")
        
    except Exception as e:
        logger.error(f"Error storing formatted result: {str(e)}")

def send_formatting_metrics(data_type: str, model: str, result: Dict[str, Any]) -> None:
    """
    Send formatting metrics to CloudWatch.
    
    Args:
        data_type: Type of data formatted
        model: Model used
        result: Formatting result
    """
    try:
        metrics = []
        
        # Success/failure metrics
        metrics.append({
            "MetricName": "RealTimeFormattingSuccess",
            "Value": 1 if result.get("success") else 0,
            "Unit": "Count",
            "Dimensions": [
                {"Name": "DataType", "Value": data_type},
                {"Name": "Model", "Value": model}
            ]
        })
        
        # Processing time metrics
        processing_time = result.get("processing_time_ms", 0)
        metrics.append({
            "MetricName": "RealTimeFormattingLatency",
            "Value": processing_time,
            "Unit": "Milliseconds",
            "Dimensions": [
                {"Name": "DataType", "Value": data_type},
                {"Name": "Model", "Value": model}
            ]
        })
        
        # Cache hit/miss metrics
        cached = result.get("cached", False)
        metrics.append({
            "MetricName": "RealTimeFormattingCacheHit",
            "Value": 1 if cached else 0,
            "Unit": "Count",
            "Dimensions": [
                {"Name": "DataType", "Value": data_type}
            ]
        })
        
        # Send metrics to CloudWatch
        cloudwatch_client.put_metric_data(
            Namespace="CustomerFeedback/RealTimeFormatting",
            MetricData=metrics
        )
        
        logger.info(f"Sent {len(metrics)} real-time formatting metrics to CloudWatch")
        
    except Exception as e:
        logger.error(f"Error sending formatting metrics: {str(e)}")

def send_timing_metrics(processing_time_ms: float) -> None:
    """
    Send Lambda timing metrics to CloudWatch.
    
    Args:
        processing_time_ms: Processing time in milliseconds
    """
    try:
        cloudwatch_client.put_metric_data(
            Namespace="CustomerFeedback/RealTimeFormatting",
            MetricData=[{
                "MetricName": "LambdaExecutionTime",
                "Value": processing_time_ms,
                "Unit": "Milliseconds"
            }]
        )
        
    except Exception as e:
        logger.error(f"Error sending timing metrics: {str(e)}")

# Warm up function for provisioned concurrency
def warm_up() -> None:
    """
    Warm up function for provisioned concurrency.
    """
    logger.info("Warming up real-time formatter")
    
    # Initialize formatters
    try:
        text_formatter = get_formatter("text", "claude-v2")
        image_formatter = get_formatter("image", "claude-v2")
        audio_formatter = get_formatter("audio", "claude-v2")
        survey_formatter = get_formatter("survey", "claude-v2")
        
        logger.info("Formatters warmed up successfully")
        
    except Exception as e:
        logger.error(f"Error during warm up: {str(e)}")

# Initialize on cold start
if not format_cache:
    warm_up()

# Export for testing
if __name__ == "__main__":
    # Test event
    test_event = {
        "httpMethod": "POST",
        "body": json.dumps({
            "data_type": "text",
            "data": {
                "original_text": "This product is amazing! Great quality and fast shipping.",
                "sentiment": {"Sentiment": "POSITIVE", "Score": 0.92},
                "metadata": {
                    "customer_id": "CUST-00001",
                    "product_id": "PROD-12345"
                }
            },
            "model": "claude-v2",
            "format_type": "conversation",
            "use_cache": True
        })
    }
    
    # Test handler
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))