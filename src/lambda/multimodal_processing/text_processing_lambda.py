#!/usr/bin/env python3
"""
Lambda function for processing text feedback with Amazon Comprehend.

This function processes text feedback to extract sentiment, entities, and key phrases
using Amazon Comprehend. Enhanced for Phase 2 with integration to data validation layer,
improved error handling, and retry logic.
"""

import json
import boto3
import os
import logging
import re
import time
import sys
from datetime import datetime
from urllib.parse import unquote_plus
from botocore.exceptions import ClientError

# Import unified quality score calculator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.quality_score_calculator import calculate_unified_quality_score, log_quality_metrics

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Initialize AWS clients outside handler for reuse
comprehend = boto3.client('comprehend')
s3_client = boto3.client('s3')
cloudwatch = boto3.client('cloudwatch')
sqs = boto3.client('sqs')

# Environment variables for Phase 2 processing
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')
PROCESSED_BUCKET = os.environ.get('PROCESSED_BUCKET', 'lm-ai-feedback-dev')
QUALITY_THRESHOLD = float(os.environ.get('QUALITY_THRESHOLD', '0.7'))
DLQ_URL = os.environ.get('DLQ_URL', '')
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', '3'))

def lambda_handler(event, context):
    """
    Lambda handler for text processing with enhanced validation integration.
    
    Supports two trigger types:
    1. S3 event trigger (for validated data from Phase 1)
    2. API Gateway direct invocation (for real-time processing)
    
    Args:
        event (dict): S3 event or API Gateway request
        context (dict): Lambda context
        
    Returns:
        dict: Response with processing results
    """
    
    # Track execution time for performance monitoring
    start_time = datetime.now()
    request_id = context.aws_request_id
    
    logger.info(f"Starting text processing - Request ID: {request_id}")
    
    try:
        # Determine trigger type and process accordingly
        if 'Records' in event and 's3' in event['Records'][0]:
            # S3 event trigger from validated data
            return process_validated_s3_event(event, context, request_id)
        elif 'body' in event:
            # API Gateway direct invocation
            return process_api_request(event, context, request_id)
        else:
            error_msg = "Unsupported event format"
            logger.error(f"{error_msg} - Request ID: {request_id}")
            return create_response(400, {"error": error_msg, "request_id": request_id})
            
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(f"{error_msg} - Request ID: {request_id}")
        send_error_metrics('Text', request_id)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        send_execution_time_metric(execution_time, False, request_id)
        
        # Send to DLQ if configured
        if DLQ_URL:
            send_to_dlq(event, str(e), request_id)
        
        return create_response(500, {"error": error_msg, "request_id": request_id})

def process_validated_s3_event(event, context, request_id):
    """
    Process validated text file from S3 event trigger with integration to Phase 1 validation.
    
    Args:
        event (dict): S3 event
        context (dict): Lambda context
        request_id (str): Lambda request ID for tracking
        
    Returns:
        dict: Response with processing results
    """
    
    # Get the S3 object
    record = event['Records'][0]['s3']
    bucket = record['bucket']['name']
    key = unquote_plus(record['object']['key'])
    
    logger.info(f"Processing validated S3 object: {bucket}/{key} - Request ID: {request_id}")
    
    # Process validated data from Phase 1 (expected format: processed/text_reviews/filename_validated.json)
    if not key.startswith('processed/text_reviews/') or not key.endswith('_validated.json'):
        logger.info(f"Skipping non-validated text file: {key} - Request ID: {request_id}")
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Not a validated text review file', 'request_id': request_id})
        }
    
    # Get the validated data with retry logic
    try:
        validated_data = retry_on_failure(
            s3_client.get_object,
            max_retries=MAX_RETRIES,
            Bucket=bucket,
            Key=key
        )
        validated_content = json.loads(validated_data['Body'].read().decode('utf-8'))
        
        # Extract text content and validation metadata
        text_content = validated_content.get('text_content', '')
        validation_results = validated_content.get('validation_results', {})
        customer_id = validated_content.get('customer_id', extract_customer_id_from_filename(key))
        
        # Check validation quality score
        validation_score = validation_results.get('quality_score', 0.0)
        if validation_score < QUALITY_THRESHOLD:
            logger.warning(f"Low validation score ({validation_score}) for {key} - Request ID: {request_id}")
            # Still process but flag for review
        
        # Process the text with enhanced analysis
        processed_result = process_text_content(text_content, customer_id, request_id)
        
        # Combine validation and processing results
        combined_result = combine_validation_and_processing_results(
            validated_data, processed_result, request_id
        )
        
        # Save processed results with enhanced metadata
        processed_key = key.replace('_validated.json', '_processed.json')
        
        retry_on_failure(
            s3_client.put_object,
            max_retries=MAX_RETRIES,
            Bucket=PROCESSED_BUCKET,
            Key=processed_key,
            Body=json.dumps(combined_result, default=str),
            ContentType='application/json',
            Metadata={
                'request_id': request_id,
                'validation_score': str(validation_score),
                'processing_timestamp': datetime.now().isoformat()
            }
        )
        
        # Send success metrics
        send_processing_metrics(
            'Text',
            len(text_content),
            len(processed_result.get('entities', [])),
            request_id
        )
        
        logger.info(f"Successfully processed validated text review: {key} - Request ID: {request_id}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully processed validated review',
                'processed_key': processed_key,
                'request_id': request_id
            })
        }
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"S3 ClientError processing {key}: {error_code} - {str(e)} - Request ID: {request_id}")
        send_error_metrics('Text', request_id)
        
        # Send to DLQ if configured
        if DLQ_URL:
            send_to_dlq(event, f"S3 ClientError: {error_code} - {str(e)}", request_id)
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f"S3 error: {error_code}", 'request_id': request_id})
        }
    except Exception as e:
        logger.error(f"Error processing validated S3 object {key}: {str(e)} - Request ID: {request_id}")
        send_error_metrics('Text', request_id)
        
        # Send to DLQ if configured
        if DLQ_URL:
            send_to_dlq(event, str(e), request_id)
        
        raise

def process_api_request(event, context, request_id):
    """
    Process text from API Gateway request with enhanced validation and error handling.
    
    Args:
        event (dict): API Gateway request
        context (dict): Lambda context
        request_id (str): Lambda request ID for tracking
        
    Returns:
        dict: Response with processing results
    """
    
    # Parse request body
    try:
        body = json.loads(event.get('body', '{}'))
        
        # Validate required fields for UI integration
        validation_result = validate_api_request(body)
        if not validation_result['valid']:
            return create_response(400, {"error": validation_result['message'], "request_id": request_id})
        
        # Extract and validate fields
        customer_id = body.get('customerId', '').strip()
        rating = body.get('rating', 0)
        feedback_text = body.get('feedback', '').strip()
        timestamp = body.get('timestamp', datetime.now().isoformat())
        
        # Validate rating range
        if not (1 <= rating <= 5):
            return create_response(400, {"error": "Rating must be between 1 and 5", "request_id": request_id})
        
        # Validate feedback length
        if len(feedback_text) < 10:
            return create_response(400, {"error": "Feedback must be at least 10 characters long", "request_id": request_id})
        
        if len(feedback_text) > 500:
            return create_response(400, {"error": "Feedback cannot exceed 500 characters", "request_id": request_id})
        
        # Validate customer ID format
        if not re.match(r'^[A-Za-z0-9\-_]+$', customer_id):
            return create_response(400, {"error": "Customer ID contains invalid characters", "request_id": request_id})
        
        # Process the text with enhanced analysis
        processed_result = process_text_content(feedback_text, customer_id, request_id)
        
        # Add UI-specific metadata
        processed_result['ui_metadata'] = {
            'rating': rating,
            'submission_timestamp': timestamp,
            'source': 'ui_form',
            'environment': ENVIRONMENT,
            'request_id': request_id
        }
        
        # Store in S3 for audit trail with structured naming
        submission_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"processed/api_feedback/{customer_id}_{submission_time}_processed.json"
        
        # Add retry logic for S3 upload
        retry_on_failure(
            s3_client.put_object,
            max_retries=MAX_RETRIES,
            Bucket=PROCESSED_BUCKET,
            Key=s3_key,
            Body=json.dumps(processed_result, default=str),
            ContentType='application/json',
            Metadata={
                'request_id': request_id,
                'source': 'ui_form',
                'customer_id': customer_id,
                'rating': str(rating)
            }
        )
        
        # Send success metrics
        send_processing_metrics('API', len(feedback_text), len(processed_result.get('entities', [])), request_id)
        send_ui_metrics(rating, len(feedback_text), request_id)
        
        logger.info(f"Successfully processed API request from customer: {customer_id} - Request ID: {request_id}")
        
        # Return success response with processing results
        return create_response(200, {
            "success": True,
            "message": "Feedback processed successfully",
            "results": {
                "customer_id": customer_id,
                "sentiment": processed_result.get('sentiment'),
                "sentiment_scores": processed_result.get('sentiment_scores'),
                "key_phrases": processed_result.get('key_phrases', [])[:5],  # Limit for UI
                "processing_id": f"{customer_id}_{submission_time}",
                "request_id": request_id
            }
        })
        
    except json.JSONDecodeError:
        return create_response(400, {"error": "Invalid JSON in request body", "request_id": request_id})
    except Exception as e:
        logger.error(f"Error processing API request: {str(e)} - Request ID: {request_id}")
        send_error_metrics('API', request_id)
        return create_response(500, {"error": "Internal processing error", "request_id": request_id})

def process_text_content(text_content, customer_id, request_id):
    """
    Process text content using Amazon Comprehend with enhanced analysis and error handling.
    
    Args:
        text_content (str): Text to process
        customer_id (str): Customer ID for metadata
        request_id (str): Request ID for tracking
        
    Returns:
        dict: Processed results with sentiment, entities, and key phrases
    """
    
    # Clean and normalize text
    cleaned_text = clean_text(text_content)
    
    # Use Amazon Comprehend for analysis with retry logic
    results = {}
    
    # Detect sentiment (core analysis)
    try:
        sentiment_response = retry_on_failure(
            comprehend.detect_sentiment,
            max_retries=MAX_RETRIES,
            Text=cleaned_text,
            LanguageCode='en'
        )
        results['sentiment'] = sentiment_response['Sentiment']
        results['sentiment_scores'] = sentiment_response['SentimentScore']
    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {str(e)} - Request ID: {request_id}")
        results['sentiment'] = 'UNKNOWN'
        results['sentiment_scores'] = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Mixed': 0}
    
    # Detect key phrases (cost-effective alternative to full entity extraction)
    try:
        key_phrases_response = retry_on_failure(
            comprehend.detect_key_phrases,
            max_retries=MAX_RETRIES,
            Text=cleaned_text,
            LanguageCode='en'
        )
        results['key_phrases'] = key_phrases_response['KeyPhrases']
    except Exception as e:
        logger.warning(f"Key phrase detection failed: {str(e)} - Request ID: {request_id}")
        results['key_phrases'] = []
    
    # Enhanced entity extraction with more comprehensive analysis
    try:
        entity_response = retry_on_failure(
            comprehend.detect_entities,
            max_retries=MAX_RETRIES,
            Text=cleaned_text,
            LanguageCode='en'
        )
        # Filter to only most important entity types for cost optimization
        important_entities = ['ORGANIZATION', 'PERSON', 'LOCATION', 'PRODUCT', 'EVENT', 'COMMERCIAL_ITEM']
        results['entities'] = [
            entity for entity in entity_response['Entities']
            if entity['Type'] in important_entities
        ]
    except Exception as e:
        logger.warning(f"Entity extraction failed: {str(e)} - Request ID: {request_id}")
        results['entities'] = []
    
    # Add enhanced metadata
    results['metadata'] = {
        'customer_id': customer_id,
        'text_length': len(cleaned_text),
        'processed_timestamp': datetime.now().isoformat(),
        'environment': ENVIRONMENT,
        'request_id': request_id,
        'processing_version': '2.0'  # Phase 2 version
    }
    
    # Add insights for better analysis
    results['insights'] = extract_insights(results)
    
    return results

def clean_text(text):
    """
    Clean and normalize text for processing.
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Cleaned text
    """
    
    # Remove excessive whitespace and normalize
    cleaned = ' '.join(text.split())
    
    # Remove any non-printable characters
    cleaned = ''.join(char for char in cleaned if char.isprintable() or char.isspace())
    
    return cleaned.strip()

def extract_customer_id_from_filename(filename):
    """
    Extract customer ID from filename.
    
    Args:
        filename (str): S3 object filename
        
    Returns:
        str: Customer ID
    """
    
    try:
        # Extract from pattern: text_reviews/review_CUST-XXXXX.txt
        parts = filename.split('/')
        if len(parts) >= 2:
            file_part = parts[1]  # review_CUST-XXXXX.txt
            if file_part.startswith('review_'):
                return file_part.replace('review_', '').replace('.txt', '')
        return 'unknown'
    except Exception:
        return 'unknown'

def extract_insights(processed_result):
    """
    Extract additional insights from processed text.
    
    Args:
        processed_result (dict): Processed text results
        
    Returns:
        dict: Additional insights
    """
    
    insights = {}
    
    # Sentiment analysis insights
    sentiment_scores = processed_result.get('sentiment_scores', {})
    if sentiment_scores:
        dominant_sentiment = processed_result.get('sentiment', 'NEUTRAL')
        confidence = max(sentiment_scores.values()) if sentiment_scores else 0
        
        insights['sentiment_analysis'] = {
            'dominant_sentiment': dominant_sentiment,
            'confidence': round(confidence, 3),
            'is_strong_sentiment': confidence > 0.8,
            'is_mixed_sentiment': sentiment_scores.get('Mixed', 0) > 0.3
        }
    
    # Key phrase insights
    key_phrases = processed_result.get('key_phrases', [])
    if key_phrases:
        # Extract top phrases by score
        sorted_phrases = sorted(key_phrases, key=lambda x: x.get('Score', 0), reverse=True)
        insights['key_themes'] = [phrase['Text'] for phrase in sorted_phrases[:5]]
        
        # Categorize phrases
        insights['phrase_categories'] = categorize_phrases(sorted_phrases)
    
    # Entity insights
    entities = processed_result.get('entities', [])
    if entities:
        entity_types = {}
        for entity in entities:
            entity_type = entity['Type']
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity['Text'])
        
        insights['entity_summary'] = {
            'total_entities': len(entities),
            'entity_types': list(entity_types.keys()),
            'entity_categories': entity_types
        }
    
    return insights

def categorize_phrases(phrases):
    """
    Categorize key phrases into themes.
    
    Args:
        phrases (list): List of key phrases
        
    Returns:
        dict: Categorized phrases
    """
    
    categories = {
        'product_quality': [],
        'customer_service': [],
        'delivery_shipping': [],
        'price_value': [],
        'general': []
    }
    
    # Simple keyword-based categorization
    quality_keywords = ['quality', 'product', 'material', 'design', 'performance', 'feature']
    service_keywords = ['service', 'support', 'help', 'staff', 'representative', 'team']
    delivery_keywords = ['delivery', 'shipping', 'package', 'arrived', 'fast', 'slow']
    price_keywords = ['price', 'cost', 'value', 'expensive', 'cheap', 'worth', 'money']
    
    for phrase in phrases[:10]:  # Limit for cost optimization
        phrase_text = phrase['Text'].lower()
        
        if any(keyword in phrase_text for keyword in quality_keywords):
            categories['product_quality'].append(phrase['Text'])
        elif any(keyword in phrase_text for keyword in service_keywords):
            categories['customer_service'].append(phrase['Text'])
        elif any(keyword in phrase_text for keyword in delivery_keywords):
            categories['delivery_shipping'].append(phrase['Text'])
        elif any(keyword in phrase_text for keyword in price_keywords):
            categories['price_value'].append(phrase['Text'])
        else:
            categories['general'].append(phrase['Text'])
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

def create_response(status_code, body):
    """
    Create API Gateway response.
    
    Args:
        status_code (int): HTTP status code
        body (dict): Response body
        
    Returns:
        dict: API Gateway response format
    """
    
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'  # CORS support
        },
        'body': json.dumps(body)
    }

def send_processing_metrics(data_type, text_length, entity_count):
    """
    Send processing metrics to CloudWatch for cost monitoring.
    
    Args:
        data_type (str): Type of data processed (S3 or API)
        text_length (int): Length of text processed
        entity_count (int): Number of entities extracted
    """
    
    try:
        cloudwatch.put_metric_data(
            Namespace='CustomerFeedback/Processing',
            MetricData=[
                {
                    'MetricName': 'ProcessedCount',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'DataType', 'Value': data_type},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                },
                {
                    'MetricName': 'TextLength',
                    'Value': text_length,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'DataType', 'Value': data_type},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                },
                {
                    'MetricName': 'EntityCount',
                    'Value': entity_count,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'DataType', 'Value': data_type},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                }
            ]
        )
    except Exception as e:
        logger.warning(f"Failed to send metrics: {str(e)}")

def send_error_metrics(data_type, request_id):
    """
    Send error metrics to CloudWatch.
    
    Args:
        data_type (str): Type of data that failed to process
        request_id (str): Request ID for tracking
    """
    
    try:
        cloudwatch.put_metric_data(
            Namespace='CustomerFeedback/Processing',
            MetricData=[
                {
                    'MetricName': 'ProcessingErrors',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'DataType', 'Value': data_type},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                }
            ]
        )
    except Exception as e:
        logger.warning(f"Failed to send error metrics: {str(e)}")

def send_execution_time_metric(execution_time, success):
    """
    Send execution time metric for performance monitoring.
    
    Args:
        execution_time (float): Execution time in seconds
        success (bool): Whether processing was successful
    """
    
    try:
        cloudwatch.put_metric_data(
            Namespace='CustomerFeedback/Processing',
            MetricData=[
                {
                    'MetricName': 'ExecutionTime',
                    'Value': execution_time,
                    'Unit': 'Seconds',
                    'Dimensions': [
                        {'Name': 'Success', 'Value': 'True' if success else 'False'},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                }
            ]
        )
    except Exception as e:
        logger.warning(f"Failed to send execution time metrics: {str(e)}")

def validate_api_request(body):
    """
    Validate API request body for UI integration.
    
    Args:
        body (dict): Request body to validate
        
    Returns:
        dict: Validation result with 'valid' and 'message' keys
    """
    
    required_fields = ['customerId', 'rating', 'feedback']
    
    # Check required fields
    for field in required_fields:
        if field not in body or body[field] is None:
            return {
                'valid': False,
                'message': f'Missing required field: {field}'
            }
    
    # Validate field types
    if not isinstance(body.get('customerId'), str):
        return {'valid': False, 'message': 'Customer ID must be a string'}
    
    if not isinstance(body.get('rating'), int):
        return {'valid': False, 'message': 'Rating must be an integer'}
    
    if not isinstance(body.get('feedback'), str):
        return {'valid': False, 'message': 'Feedback must be a string'}
    
    return {'valid': True, 'message': 'Valid request'}

def send_ui_metrics(rating, feedback_length):
    """
    Send UI-specific metrics to CloudWatch.
    
    Args:
        rating (int): Customer rating (1-5)
        feedback_length (int): Length of feedback text
    """
    
    try:
        cloudwatch.put_metric_data(
            Namespace='CustomerFeedback/UI',
            MetricData=[
                {
                    'MetricName': 'RatingDistribution',
                    'Value': rating,
                    'Unit': 'None',
                    'Dimensions': [
                        {'Name': 'Rating', 'Value': str(rating)},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                },
                {
                    'MetricName': 'FeedbackLength',
                    'Value': feedback_length,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                }
            ]
        )
    except Exception as e:
        logger.warning(f"Failed to send UI metrics: {str(e)}")

def combine_validation_and_processing_results(validated_data, processed_result, request_id):
    """
    Combine validation results from Phase 1 with processing results from Phase 2.
    
    Args:
        validated_data (dict): Validation results from Phase 1
        processed_result (dict): Processing results from Phase 2
        request_id (str): Request ID for tracking
        
    Returns:
        dict: Combined results with enhanced metadata
    """
    
    validation_results = validated_data.get('validation_results', {})
    processing_results = processed_result.get('insights', {})
    
    combined_result = {
        'validation_metadata': validation_results,
        'processing_results': processed_result,
        'combined_quality_score': calculate_unified_quality_score(
            validation_results, processed_result
        ),
        'processing_timestamp': datetime.now().isoformat(),
        'request_id': request_id,
        'customer_id': validated_data.get('customer_id', processed_result['metadata']['customer_id']),
        'text_content': validated_data.get('text_content', ''),
        'sentiment': processed_result.get('sentiment'),
        'sentiment_scores': processed_result.get('sentiment_scores'),
        'key_phrases': processed_result.get('key_phrases', []),
        'entities': processed_result.get('entities', []),
        'metadata': {
            'validation_version': '1.0',
            'processing_version': '2.0',
            'environment': ENVIRONMENT,
            'integration_phase': 'phase2'
        }
    }
    
    return combined_result

def calculate_combined_quality_score(validation_results, processing_results):
    """
    Calculate combined quality score from validation and processing results.
    
    Args:
        validation_results (dict): Validation results from Phase 1
        processing_results (dict): Processing results from Phase 2
        
    Returns:
        float: Combined quality score (0-1)
    """
    
    validation_score = validation_results.get('quality_score', 0.0)
    
    # Calculate processing quality score based on processing results
    processing_score = 0.0
    
    # Check if sentiment was detected successfully
    if processing_results.get('sentiment') != 'UNKNOWN':
        processing_score += 0.3
    
    # Check if entities were extracted
    if processing_results.get('entity_summary', {}).get('total_entities', 0) > 0:
        processing_score += 0.3
    
    # Check if key themes were identified
    if processing_results.get('key_themes'):
        processing_score += 0.2
    
    # Check if sentiment analysis has good confidence
    sentiment_analysis = processing_results.get('sentiment_analysis', {})
    if sentiment_analysis.get('confidence', 0) > 0.7:
        processing_score += 0.2
    
    # Weighted average (70% validation, 30% processing)
    combined_score = (validation_score * 0.7) + (processing_score * 0.3)
    return round(min(combined_score, 1.0), 3)  # Cap at 1.0

def send_to_dlq(event, error_message, request_id):
    """
    Send failed event to Dead Letter Queue for retry processing.
    
    Args:
        event (dict): Original event that failed
        error_message (str): Error message
        request_id (str): Request ID for tracking
    """
    
    if not DLQ_URL:
        logger.warning("DLQ URL not configured, skipping DLQ send")
        return
    
    try:
        dlq_message = {
            'original_event': event,
            'error_message': error_message,
            'request_id': request_id,
            'failed_timestamp': datetime.now().isoformat(),
            'lambda_function': 'TextProcessingFunctionLM'
        }
        
        retry_on_failure(
            sqs.send_message,
            max_retries=MAX_RETRIES,
            QueueUrl=DLQ_URL,
            MessageBody=json.dumps(dlq_message),
            MessageAttributes={
                'RequestID': {
                    'DataType': 'String',
                    'StringValue': request_id
                },
                'ErrorType': {
                    'DataType': 'String',
                    'StringValue': 'ProcessingFailure'
                }
            }
        )
        
        logger.info(f"Sent failed event to DLQ - Request ID: {request_id}")
        
    except Exception as e:
        logger.error(f"Failed to send to DLQ: {str(e)} - Request ID: {request_id}")

def send_processing_metrics(data_type, text_length, entity_count, request_id):
    """
    Send processing metrics to CloudWatch for performance monitoring.
    
    Args:
        data_type (str): Type of data processed (S3 or API)
        text_length (int): Length of text processed
        entity_count (int): Number of entities extracted
        request_id (str): Request ID for tracking
    """
    
    try:
        cloudwatch.put_metric_data(
            Namespace='CustomerFeedback/ProcessingQuality',
            MetricData=[
                {
                    'MetricName': 'ProcessedCount',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'DataType', 'Value': data_type},
                        {'Name': 'Environment', 'Value': ENVIRONMENT},
                        {'Name': 'Function', 'Value': 'TextProcessing'}
                    ]
                },
                {
                    'MetricName': 'TextLength',
                    'Value': text_length,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'DataType', 'Value': data_type},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                },
                {
                    'MetricName': 'EntityCount',
                    'Value': entity_count,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'DataType', 'Value': data_type},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                },
                {
                    'MetricName': 'ProcessingQualityScore',
                    'Value': calculate_processing_quality_score(data_type, text_length, entity_count),
                    'Unit': 'None',
                    'Dimensions': [
                        {'Name': 'DataType', 'Value': data_type},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                }
            ]
        )
    except Exception as e:
        logger.warning(f"Failed to send processing metrics: {str(e)} - Request ID: {request_id}")

def calculate_processing_quality_score(data_type, text_length, entity_count):
    """
    Calculate processing quality score based on processing results.
    
    Args:
        data_type (str): Type of data processed
        text_length (int): Length of text processed
        entity_count (int): Number of entities extracted
        
    Returns:
        float: Processing quality score (0-1)
    """
    
    score = 0.0
    
    # Base score for successful processing
    score += 0.4
    
    # Bonus for entity extraction
    if entity_count > 0:
        score += min(0.3, entity_count * 0.1)
    
    # Bonus for text length (longer texts provide more context)
    if text_length > 50:
        score += 0.2
    
    # Bonus for data type
    if data_type == 'API':
        score += 0.1  # API requests are typically higher quality
    
    return min(score, 1.0)




def retry_on_failure(func, max_retries=3, delay=1.0, *args, **kwargs):
    """
    Retry function on failure with exponential backoff and enhanced logging.
    
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
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if attempt == max_retries:
                logger.error(f"Final attempt failed with ClientError {error_code}: {str(e)}")
                raise e
            
            wait_time = delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed with ClientError {error_code}, retrying in {wait_time:.2f} seconds")
            time.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Final attempt failed: {str(e)}")
                raise e
            
            wait_time = delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time:.2f} seconds: {str(e)}")
            time.sleep(wait_time)

if __name__ == "__main__":
    # For local testing
    test_event = {
        'Records': [
            {
                's3': {
                    'bucket': {'name': 'test-bucket'},
                    'object': {'key': 'text_reviews/review_CUST-00001.txt'}
                }
            }
        ]
    }
    
    # Test API request
    api_test_event = {
        'body': json.dumps({
            'customerId': 'CUST-00001',
            'rating': 5,
            'feedback': 'Excellent service and very helpful staff!',
            'timestamp': datetime.now().isoformat()
        })
    }
    
    # This would be used for testing with actual S3 data
    # lambda_handler(test_event, None)
    # lambda_handler(api_test_event, None)