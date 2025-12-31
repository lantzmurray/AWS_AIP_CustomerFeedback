#!/usr/bin/env python3
"""
Common utility functions used across the AWS data validation and processing pipeline.

This module contains shared helper functions for data processing,
AWS service interactions, and common operations.
"""

import json
import boto3
import os
import re
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# AWS service clients
_s3_client = None
_cloudwatch_client = None
_comprehend_client = None

def get_s3_client():
    """Get or create S3 client."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3')
    return _s3_client

def get_cloudwatch_client():
    """Get or create CloudWatch client."""
    global _cloudwatch_client
    if _cloudwatch_client is None:
        _cloudwatch_client = boto3.client('cloudwatch')
    return _cloudwatch_client

def get_comprehend_client():
    """Get or create Comprehend client."""
    global _comprehend_client
    if _comprehend_client is None:
        _comprehend_client = boto3.client('comprehend')
    return _comprehend_client

def read_json_from_s3(bucket: str, key: str) -> Dict[str, Any]:
    """
    Read JSON file from S3.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        
    Returns:
        dict: Parsed JSON data
    """
    try:
        s3_client = get_s3_client()
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"Error reading {key} from S3: {str(e)}")
        return {}

def write_json_to_s3(bucket: str, key: str, data: Dict[str, Any], 
                     content_type: str = 'application/json') -> bool:
    """
    Write JSON data to S3.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        data (dict): Data to write
        content_type (str): Content type header
        
    Returns:
        bool: Success status
    """
    try:
        s3_client = get_s3_client()
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, indent=2),
            ContentType=content_type
        )
        return True
    except Exception as e:
        print(f"Error writing {key} to S3: {str(e)}")
        return False

def send_metric_to_cloudwatch(namespace: str, metric_name: str, value: float, 
                          unit: str = 'None', dimensions: Optional[List[Dict]] = None) -> bool:
    """
    Send custom metric to CloudWatch.
    
    Args:
        namespace (str): CloudWatch namespace
        metric_name (str): Metric name
        value (float): Metric value
        unit (str): Metric unit
        dimensions (list): List of dimensions
        
    Returns:
        bool: Success status
    """
    try:
        cloudwatch = get_cloudwatch_client()
        
        metric_data = {
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit
        }
        
        if dimensions:
            metric_data['Dimensions'] = dimensions
        
        cloudwatch.put_metric_data(
            Namespace=namespace,
            MetricData=[metric_data]
        )
        return True
    except Exception as e:
        print(f"Error sending metric {metric_name}: {str(e)}")
        return False

def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email (str): Email address to validate
        
    Returns:
        bool: True if valid email format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe S3 storage.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    # Ensure it's not empty
    if not filename:
        filename = f"unnamed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return filename

def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename (str): Filename
        
    Returns:
        str: File extension (lowercase, without dot)
    """
    return os.path.splitext(filename)[1][1:].lower()

def is_text_file(filename: str) -> bool:
    """
    Check if file is a text file.
    
    Args:
        filename (str): Filename to check
        
    Returns:
        bool: True if text file
    """
    text_extensions = ['.txt', '.json', '.csv', '.md', '.log']
    return get_file_extension(filename) in [ext[1:] for ext in text_extensions]

def is_image_file(filename: str) -> bool:
    """
    Check if file is an image file.
    
    Args:
        filename (str): Filename to check
        
    Returns:
        bool: True if image file
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    return get_file_extension(filename) in [ext[1:] for ext in image_extensions]

def is_audio_file(filename: str) -> bool:
    """
    Check if file is an audio file.
    
    Args:
        filename (str): Filename to check
        
    Returns:
        bool: True if audio file
    """
    audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
    return get_file_extension(filename) in [ext[1:] for ext in audio_extensions]

def calculate_quality_score(checks: Dict[str, bool]) -> float:
    """
    Calculate quality score from validation checks.
    
    Args:
        checks (dict): Dictionary of validation check results
        
    Returns:
        float: Quality score between 0 and 1
    """
    if not checks:
        return 0.0
    
    passed_checks = sum(1 for check in checks.values() if check)
    total_checks = len(checks)
    
    return passed_checks / total_checks

def format_timestamp(timestamp: Optional[str] = None) -> str:
    """
    Format timestamp in ISO format.
    
    Args:
        timestamp (str): Optional timestamp string
        
    Returns:
        str: Formatted timestamp
    """
    if timestamp is None:
        return datetime.now().isoformat()
    
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.isoformat()
    except:
        return datetime.now().isoformat()

def get_environment_variable(name: str, default_value: str = '') -> str:
    """
    Get environment variable with default value.
    
    Args:
        name (str): Environment variable name
        default_value (str): Default value if not set
        
    Returns:
        str: Environment variable value or default
    """
    return os.environ.get(name, default_value)

def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        suffix (str): Suffix to add if truncated
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def extract_entities_from_text(text: str, comprehend_client=None) -> List[Dict]:
    """
    Extract entities from text using Amazon Comprehend.
    
    Args:
        text (str): Text to analyze
        comprehend_client: Optional Comprehend client
        
    Returns:
        list: List of entities
    """
    try:
        if comprehend_client is None:
            comprehend_client = get_comprehend_client()
        
        response = comprehend_client.detect_entities(
            Text=text,
            LanguageCode='en'
        )
        
        return response.get('Entities', [])
    except Exception as e:
        print(f"Error extracting entities: {str(e)}")
        return []

def detect_sentiment(text: str, comprehend_client=None) -> Dict[str, Any]:
    """
    Detect sentiment from text using Amazon Comprehend.
    
    Args:
        text (str): Text to analyze
        comprehend_client: Optional Comprehend client
        
    Returns:
        dict: Sentiment analysis results
    """
    try:
        if comprehend_client is None:
            comprehend_client = get_comprehend_client()
        
        response = comprehend_client.detect_sentiment(
            Text=text,
            LanguageCode='en'
        )
        
        return {
            'sentiment': response.get('Sentiment'),
            'sentiment_scores': response.get('SentimentScore', {})
        }
    except Exception as e:
        print(f"Error detecting sentiment: {str(e)}")
        return {'sentiment': 'UNKNOWN', 'sentiment_scores': {}}

def create_error_response(error_message: str, error_code: int = 500) -> Dict[str, Any]:
    """
    Create standardized error response.
    
    Args:
        error_message (str): Error message
        error_code (int): HTTP error code
        
    Returns:
        dict: Error response
    """
    return {
        'statusCode': error_code,
        'body': json.dumps({
            'error': True,
            'message': error_message,
            'timestamp': format_timestamp()
        }),
        'headers': {
            'Content-Type': 'application/json'
        }
    }

def create_success_response(data: Dict[str, Any], message: str = 'Success') -> Dict[str, Any]:
    """
    Create standardized success response.
    
    Args:
        data (dict): Response data
        message (str): Success message
        
    Returns:
        dict: Success response
    """
    return {
        'statusCode': 200,
        'body': json.dumps({
            'error': False,
            'message': message,
            'data': data,
            'timestamp': format_timestamp()
        }),
        'headers': {
            'Content-Type': 'application/json'
        }
    }

def log_lambda_event(event: Dict[str, Any], context: Any) -> None:
    """
    Log Lambda event and context information.
    
    Args:
        event (dict): Lambda event
        context: Lambda context
    """
    print(f"Lambda function: {context.function_name}")
    print(f"Lambda version: {context.function_version}")
    print(f"Request ID: {context.aws_request_id}")
    print(f"Event: {json.dumps(event, indent=2)}")

def get_s3_object_size(bucket: str, key: str) -> int:
    """
    Get S3 object size in bytes.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        
    Returns:
        int: Object size in bytes
    """
    try:
        s3_client = get_s3_client()
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return response.get('ContentLength', 0)
    except Exception as e:
        print(f"Error getting object size: {str(e)}")
        return 0

def is_valid_json(text: str) -> bool:
    """
    Check if text is valid JSON.
    
    Args:
        text (str): Text to validate
        
    Returns:
        bool: True if valid JSON
    """
    try:
        json.loads(text)
        return True
    except:
        return False

def parse_s3_event(event: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Parse S3 event to extract bucket and key information.
    
    Args:
        event (dict): S3 event
        
    Returns:
        list: List of S3 object references
    """
    objects = []
    
    for record in event.get('Records', []):
        if record.get('eventSource') == 'aws:s3':
            s3_info = record.get('s3', {})
            bucket = s3_info.get('bucket', {}).get('name', '')
            key = s3_info.get('object', {}).get('key', '')
            
            if bucket and key:
                objects.append({'bucket': bucket, 'key': key})
    
    return objects

def retry_on_failure(func, max_retries: int = 3, delay: float = 1.0, *args, **kwargs):
    """
    Retry function on failure with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries (int): Maximum number of retries
        delay (float): Initial delay between retries
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or raises last exception
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            wait_time = delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed, retrying in {wait_time:.2f} seconds: {str(e)}")
            import time
            time.sleep(wait_time)

def get_aws_region() -> str:
    """
    Get current AWS region.
    
    Returns:
        str: AWS region name
    """
    return get_environment_variable('AWS_REGION', 'us-east-1')

def generate_uuid() -> str:
    """
    Generate a unique identifier.
    
    Returns:
        str: UUID string
    """
    import uuid
    return str(uuid.uuid4())

def calculate_date_range(start_date: str, end_date: str) -> int:
    """
    Calculate number of days between two dates.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        int: Number of days between dates
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        return abs((end - start).days)
    except Exception as e:
        print(f"Error calculating date range: {str(e)}")
        return 0

# Constants
SUPPORTED_TEXT_FORMATS = ['.txt', '.json', '.csv', '.md']
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.flac', '.aac', '.ogg']

DEFAULT_QUALITY_THRESHOLD = 0.7
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.7

if __name__ == "__main__":
    # Test some utility functions
    print("Testing common utility functions...")
    
    # Test file type detection
    print(f"Is text file: {is_text_file('test.txt')}")
    print(f"Is image file: {is_image_file('test.jpg')}")
    print(f"Is audio file: {is_audio_file('test.mp3')}")
    
    # Test quality score calculation
    checks = {'check1': True, 'check2': False, 'check3': True}
    print(f"Quality score: {calculate_quality_score(checks)}")
    
    # Test timestamp formatting
    print(f"Current timestamp: {format_timestamp()}")
    
    print("Utility function tests completed!")