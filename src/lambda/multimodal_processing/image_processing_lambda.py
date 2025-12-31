#!/usr/bin/env python3
"""
Lambda function for processing product images with Amazon Textract and Rekognition.

This function processes product images to extract text and analyze visual content
using Amazon Textract and Amazon Rekognition. Enhanced for Phase 2 with integration
to data validation layer, optimized memory usage, and improved error handling.
"""

import json
import boto3
import os
import base64
import logging
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
s3_client = boto3.client('s3')
textract = boto3.client('textract')
rekognition = boto3.client('rekognition')
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
    Lambda handler for image processing with enhanced validation integration.
    
    Args:
        event (dict): S3 event trigger
        context (dict): Lambda context
        
    Returns:
        dict: Response with processing results
    """
    
    # Track execution time for performance monitoring
    start_time = datetime.now()
    request_id = context.aws_request_id
    
    logger.info(f"Starting image processing - Request ID: {request_id}")
    
    try:
        # Get the S3 object
        record = event['Records'][0]['s3']
        bucket = record['bucket']['name']
        key = unquote_plus(record['object']['key'])
        
        logger.info(f"Processing S3 object: {bucket}/{key} - Request ID: {request_id}")
        
        # Process validated data from Phase 1 (expected format: processed/images/filename_validated.json)
        if key.startswith('processed/images/') and key.endswith('_validated.json'):
            return process_validated_image_metadata(bucket, key, request_id)
        
        # Only process image files
        if not key.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            logger.info(f"Skipping non-image file: {key} - Request ID: {request_id}")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Not an image file',
                    'request_id': request_id
                })
            }
        
        # Process raw image file
        return process_raw_image(bucket, key, request_id, start_time)
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(f"{error_msg} - Request ID: {request_id}")
        send_error_metrics('Image', request_id)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        send_execution_time_metric(execution_time, False, request_id)
        
        # Send to DLQ if configured
        if DLQ_URL:
            send_to_dlq(event, str(e), request_id)
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': error_msg, 'request_id': request_id})
        }

def process_validated_image_metadata(bucket, key, request_id):
    """
    Process validated image metadata from Phase 1.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        request_id (str): Request ID for tracking
        
    Returns:
        dict: Response with processing results
    """
    
    logger.info(f"Processing validated image metadata: {key} - Request ID: {request_id}")
    
    try:
        # Get validated data with retry logic
        validated_data = retry_on_failure(
            s3_client.get_object,
            max_retries=MAX_RETRIES,
            Bucket=bucket,
            Key=key
        )
        validated_content = json.loads(validated_data['Body'].read().decode('utf-8'))
        
        # Extract metadata and validation results
        validation_results = validated_content.get('validation_results', {})
        image_metadata = validated_content.get('image_metadata', {})
        image_key = validated_content.get('image_key', '')
        
        # Check validation quality score
        validation_score = validation_results.get('quality_score', 0.0)
        if validation_score < QUALITY_THRESHOLD:
            logger.warning(f"Low validation score ({validation_score}) for {key} - Request ID: {request_id}")
        
        # Process the actual image file
        if image_key:
            processed_result = process_image_content(bucket, image_key, request_id)
            
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
                'ValidatedImage',
                len(combined_result.get('extracted_text', '')),
                len(combined_result.get('labels', [])),
                len(combined_result.get('detected_text', [])),
                request_id
            )
            
            logger.info(f"Successfully processed validated image: {key} - Request ID: {request_id}")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Successfully processed validated image',
                    'processed_key': processed_key,
                    'request_id': request_id
                })
            }
        else:
            raise ValueError("No image key found in validated metadata")
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"S3 ClientError processing {key}: {error_code} - {str(e)} - Request ID: {request_id}")
        send_error_metrics('Image', request_id)
        
        # Send to DLQ if configured
        if DLQ_URL:
            send_to_dlq({'bucket': bucket, 'key': key}, f"S3 ClientError: {error_code} - {str(e)}", request_id)
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f"S3 error: {error_code}", 'request_id': request_id})
        }
    except Exception as e:
        logger.error(f"Error processing validated image metadata {key}: {str(e)} - Request ID: {request_id}")
        send_error_metrics('Image', request_id)
        
        # Send to DLQ if configured
        if DLQ_URL:
            send_to_dlq({'bucket': bucket, 'key': key}, str(e), request_id)
        
        raise

def process_raw_image(bucket, key, request_id, start_time):
    """
    Process raw image file directly.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        request_id (str): Request ID for tracking
        start_time (datetime): Start time for execution tracking
        
    Returns:
        dict: Response with processing results
    """
    
    logger.info(f"Processing raw image: {key} - Request ID: {request_id}")
    
    try:
        # Process the image content
        processed_result = process_image_content(bucket, key, request_id)
        
        # Add raw processing metadata
        processed_result['processing_metadata'] = {
            'source': 'raw_image',
            'validation_skipped': True,
            'environment': ENVIRONMENT,
            'request_id': request_id
        }
        
        # Save processed results
        processed_key = key.replace('images/', 'processed/images/').replace(os.path.splitext(key)[1], '_processed.json')
        
        retry_on_failure(
            s3_client.put_object,
            max_retries=MAX_RETRIES,
            Bucket=PROCESSED_BUCKET,
            Key=processed_key,
            Body=json.dumps(processed_result, default=str),
            ContentType='application/json',
            Metadata={
                'request_id': request_id,
                'source': 'raw_image',
                'processing_timestamp': datetime.now().isoformat()
            }
        )
        
        # Send success metrics
        send_processing_metrics(
            'RawImage',
            len(processed_result.get('extracted_text', '')),
            len(processed_result.get('labels', [])),
            len(processed_result.get('detected_text', [])),
            request_id
        )
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        send_execution_time_metric(execution_time, True, request_id)
        
        logger.info(f"Successfully processed raw image: {key} - Request ID: {request_id}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully processed image',
                'processed_key': processed_key,
                'request_id': request_id
            })
        }
        
    except Exception as e:
        logger.error(f"Error processing raw image {key}: {str(e)} - Request ID: {request_id}")
        send_error_metrics('Image', request_id)
        raise

def process_image_content(bucket, key, request_id):
    """
    Process image content using Textract and Rekognition with optimized memory usage.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        request_id (str): Request ID for tracking
        
    Returns:
        dict: Processed image results
    """
    
    results = {}
    
    # Extract text from the image using Amazon Textract with retry logic
    try:
        textract_response = retry_on_failure(
            textract.detect_document_text,
            max_retries=MAX_RETRIES,
            Document={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': key
                }
            }
        )
        
        # Extract the text efficiently
        extracted_text = ""
        lines = []
        for item in textract_response['Blocks']:
            if item['BlockType'] == 'LINE':
                line_text = item['Text']
                extracted_text += line_text + "\n"
                lines.append({
                    'text': line_text,
                    'confidence': item.get('Confidence', 0),
                    'bounding_box': item.get('Geometry', {}).get('BoundingBox', {})
                })
        
        results['extracted_text'] = extracted_text.strip()
        results['extracted_lines'] = lines
        
    except Exception as e:
        logger.warning(f"Textract analysis failed: {str(e)} - Request ID: {request_id}")
        results['extracted_text'] = ""
        results['extracted_lines'] = []
    
    # Analyze the image using Amazon Rekognition with optimized settings
    try:
        # Detect labels with memory optimization
        label_response = retry_on_failure(
            rekognition.detect_labels,
            max_retries=MAX_RETRIES,
            Image={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': key
                }
            },
            MaxLabels=15,  # Increased for better analysis
            MinConfidence=60  # Lowered for more comprehensive results
        )
        
        # Process labels with additional insights
        labels = []
        for label in label_response['Labels']:
            label_data = {
                'name': label['Name'],
                'confidence': label['Confidence'],
                'instances': len(label.get('Instances', [])),
                'parents': [parent['Name'] for parent in label.get('Parents', [])]
            }
            labels.append(label_data)
        
        results['labels'] = labels
        
    except Exception as e:
        logger.warning(f"Label detection failed: {str(e)} - Request ID: {request_id}")
        results['labels'] = []
    
    # Detect text (as a backup to Textract)
    try:
        text_response = retry_on_failure(
            rekognition.detect_text,
            max_retries=MAX_RETRIES,
            Image={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': key
                }
            }
        )
        
        # Process detected text
        detected_text = []
        for text in text_response['TextDetections']:
            if text['Type'] == 'LINE':
                detected_text.append({
                    'text': text['DetectedText'],
                    'confidence': text['Confidence'],
                    'bounding_box': text.get('Geometry', {}).get('BoundingBox', {})
                })
        
        results['detected_text'] = detected_text
        
    except Exception as e:
        logger.warning(f"Text detection failed: {str(e)} - Request ID: {request_id}")
        results['detected_text'] = []
    
    # Detect moderation labels (for content safety)
    try:
        moderation_response = retry_on_failure(
            rekognition.detect_moderation_labels,
            max_retries=MAX_RETRIES,
            Image={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': key
                }
            },
            MinConfidence=50  # Lower threshold for safety
        )
        
        results['moderation_labels'] = moderation_response.get('ModerationLabels', [])
        results['content_safe'] = len(results['moderation_labels']) == 0
        
    except Exception as e:
        logger.warning(f"Moderation detection failed: {str(e)} - Request ID: {request_id}")
        results['moderation_labels'] = []
        results['content_safe'] = True
    
    # Detect faces (optional, for customer feedback context)
    try:
        face_response = retry_on_failure(
            rekognition.detect_faces,
            max_retries=MAX_RETRIES,
            Image={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': key
                }
            },
            Attributes=['ALL']  # Get all facial attributes
        )
        
        face_details = face_response.get('FaceDetails', [])
        results['faces_detected'] = len(face_details)
        results['face_details'] = face_details
        
        # Add face analysis insights
        if face_details:
            results['face_analysis'] = analyze_faces(face_details)
        
    except Exception as e:
        logger.warning(f"Face detection failed: {str(e)} - Request ID: {request_id}")
        results['faces_detected'] = 0
        results['face_details'] = []
        results['face_analysis'] = {}
    
    # Add enhanced metadata
    results['metadata'] = {
        'image_key': key,
        'product_id': os.path.basename(key).split('_')[0] if '_' in os.path.basename(key) else '',
        'file_extension': os.path.splitext(key)[1].lower(),
        'processed_timestamp': datetime.now().isoformat(),
        'environment': ENVIRONMENT,
        'request_id': request_id,
        'processing_version': '2.0'  # Phase 2 version
    }
    
    # Add insights for better analysis
    results['insights'] = extract_image_insights(results)
    
    return results

def analyze_faces(face_details):
    """
    Analyze detected faces for additional insights.
    
    Args:
        face_details (list): List of face details from Rekognition
        
    Returns:
        dict: Face analysis insights
    """
    
    if not face_details:
        return {}
    
    analysis = {
        'total_faces': len(face_details),
        'gender_distribution': {},
        'age_range_distribution': {},
        'emotion_distribution': {},
        'face_quality_scores': []
    }
    
    for face in face_details:
        # Gender analysis
        gender = face.get('Gender', {})
        if gender:
            gender_value = gender.get('Value', 'Unknown')
            analysis['gender_distribution'][gender_value] = analysis['gender_distribution'].get(gender_value, 0) + 1
        
        # Age range analysis
        age_range = face.get('AgeRange', {})
        if age_range:
            age_low = age_range.get('Low', 0)
            age_high = age_range.get('High', 0)
            age_category = f"{age_low}-{age_high}"
            analysis['age_range_distribution'][age_category] = analysis['age_range_distribution'].get(age_category, 0) + 1
        
        # Emotion analysis
        emotions = face.get('Emotions', [])
        if emotions:
            # Get the dominant emotion
            dominant_emotion = max(emotions, key=lambda x: x.get('Confidence', 0))
            emotion_type = dominant_emotion.get('Type', 'Unknown')
            analysis['emotion_distribution'][emotion_type] = analysis['emotion_distribution'].get(emotion_type, 0) + 1
        
        # Face quality
        quality = face.get('Quality', {})
        if quality:
            brightness = quality.get('Brightness', 0)
            sharpness = quality.get('Sharpness', 0)
            analysis['face_quality_scores'].append({
                'brightness': brightness,
                'sharpness': sharpness,
                'overall': (brightness + sharpness) / 2
            })
    
    # Calculate average face quality
    if analysis['face_quality_scores']:
        avg_quality = sum(score['overall'] for score in analysis['face_quality_scores']) / len(analysis['face_quality_scores'])
        analysis['average_face_quality'] = round(avg_quality, 3)
    
    return analysis

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
    processing_insights = processed_result.get('insights', {})
    
    combined_result = {
        'validation_metadata': validation_results,
        'processing_results': processed_result,
        'combined_quality_score': calculate_unified_quality_score(
            validation_results, processed_result
        ),
        'processing_timestamp': datetime.now().isoformat(),
        'request_id': request_id,
        'image_key': validated_data.get('image_key', processed_result['metadata']['image_key']),
        'extracted_text': processed_result.get('extracted_text', ''),
        'labels': processed_result.get('labels', []),
        'detected_text': processed_result.get('detected_text', []),
        'moderation_labels': processed_result.get('moderation_labels', []),
        'faces_detected': processed_result.get('faces_detected', 0),
        'face_details': processed_result.get('face_details', []),
        'metadata': {
            'validation_version': '1.0',
            'processing_version': '2.0',
            'environment': ENVIRONMENT,
            'integration_phase': 'phase2'
        }
    }
    
    return combined_result

def calculate_combined_quality_score(validation_results, processed_result):
    """
    Calculate combined quality score from validation and processing results.
    
    Args:
        validation_results (dict): Validation results from Phase 1
        processed_result (dict): Processing results from Phase 2
        
    Returns:
        float: Combined quality score (0-1)
    """
    
    validation_score = validation_results.get('quality_score', 0.0)
    
    # Calculate processing quality score based on processing results
    processing_score = 0.0
    
    # Check if labels were detected
    if processed_result.get('labels'):
        processing_score += 0.3
    
    # Check if text was extracted
    if processed_result.get('extracted_text'):
        processing_score += 0.2
    
    # Check content safety
    if processed_result.get('content_safe', True):
        processing_score += 0.2
    
    # Check if dominant labels have good confidence
    dominant_labels = processed_result.get('insights', {}).get('dominant_labels', [])
    if dominant_labels and any(label.get('confidence', 0) > 80 for label in dominant_labels):
        processing_score += 0.3
    
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
            'lambda_function': 'ImageProcessingFunctionLM'
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

def send_processing_metrics(data_type, text_length, label_count, detected_text_count, request_id):
    """
    Send processing metrics to CloudWatch for performance monitoring.
    
    Args:
        data_type (str): Type of data processed
        text_length (int): Length of text extracted
        label_count (int): Number of labels detected
        detected_text_count (int): Number of text elements detected
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
                        {'Name': 'Function', 'Value': 'ImageProcessing'}
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
                    'MetricName': 'LabelCount',
                    'Value': label_count,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'DataType', 'Value': data_type},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                },
                {
                    'MetricName': 'DetectedTextCount',
                    'Value': detected_text_count,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'DataType', 'Value': data_type},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                },
                {
                    'MetricName': 'ProcessingQualityScore',
                    'Value': calculate_processing_quality_score(data_type, text_length, label_count),
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

def calculate_processing_quality_score(data_type, text_length, label_count):
    """
    Calculate processing quality score based on processing results.
    
    Args:
        data_type (str): Type of data processed
        text_length (int): Length of text extracted
        label_count (int): Number of labels extracted
        
    Returns:
        float: Processing quality score (0-1)
    """
    
    score = 0.0
    
    # Base score for successful processing
    score += 0.4
    
    # Bonus for label extraction
    if label_count > 0:
        score += min(0.3, label_count * 0.05)
    
    # Bonus for text extraction
    if text_length > 0:
        score += 0.2
    
    # Bonus for data type
    if data_type in ['ValidatedImage', 'RawImage']:
        score += 0.1  # Image processing is valuable
    
    return min(score, 1.0)

def send_error_metrics(data_type, request_id):
    """
    Send error metrics to CloudWatch with enhanced tracking.
    
    Args:
        data_type (str): Type of data that failed to process
        request_id (str): Request ID for tracking
    """
    
    try:
        cloudwatch.put_metric_data(
            Namespace='CustomerFeedback/ProcessingQuality',
            MetricData=[
                {
                    'MetricName': 'ProcessingErrors',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'DataType', 'Value': data_type},
                        {'Name': 'Environment', 'Value': ENVIRONMENT},
                        {'Name': 'Function', 'Value': 'ImageProcessing'}
                    ]
                }
            ]
        )
    except Exception as e:
        logger.warning(f"Failed to send error metrics: {str(e)} - Request ID: {request_id}")

def send_execution_time_metric(execution_time, success, request_id):
    """
    Send execution time metric for performance monitoring.
    
    Args:
        execution_time (float): Execution time in seconds
        success (bool): Whether processing was successful
        request_id (str): Request ID for tracking
    """
    
    try:
        cloudwatch.put_metric_data(
            Namespace='CustomerFeedback/ProcessingQuality',
            MetricData=[
                {
                    'MetricName': 'ExecutionTime',
                    'Value': execution_time,
                    'Unit': 'Seconds',
                    'Dimensions': [
                        {'Name': 'Success', 'Value': 'True' if success else 'False'},
                        {'Name': 'Environment', 'Value': ENVIRONMENT},
                        {'Name': 'Function', 'Value': 'ImageProcessing'}
                    ]
                }
            ]
        )
    except Exception as e:
        logger.warning(f"Failed to send execution time metrics: {str(e)} - Request ID: {request_id}")

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


def extract_image_insights(processed_image):
    """
    Extract additional insights from processed image.
    
    Args:
        processed_image (dict): Processed image data
        
    Returns:
        dict: Additional insights
    """
    
    insights = {}
    
    # Label insights
    labels = processed_image['labels']
    high_confidence_labels = [label for label in labels if label['Confidence'] > 80]
    insights['dominant_labels'] = high_confidence_labels[:5]  # Top 5 high-confidence labels
    
    # Text insights
    extracted_text = processed_image['extracted_text']
    if extracted_text.strip():
        insights['has_text'] = True
        insights['text_length'] = len(extracted_text)
        insights['word_count'] = len(extracted_text.split())
    else:
        insights['has_text'] = False
    
    # Safety insights
    moderation_labels = processed_image['moderation_labels']
    insights['content_safe'] = len(moderation_labels) == 0
    if moderation_labels:
        insights['safety_concerns'] = [label['Name'] for label in moderation_labels]
    
    # Face detection insights
    faces_detected = processed_image['faces_detected']
    insights['contains_faces'] = faces_detected > 0
    insights['face_count'] = faces_detected
    
    return insights

def get_image_for_bedrock(bucket, key):
    """
    Get image in base64 format for Bedrock multimodal requests.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        
    Returns:
        str: Base64 encoded image
    """
    
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket, Key=key)
    image_bytes = response['Body'].read()
    
    return base64.b64encode(image_bytes).decode('utf-8')

if __name__ == "__main__":
    # For local testing
    test_event = {
        'Records': [
            {
                's3': {
                    'bucket': {'name': 'test-bucket'},
                    'object': {'key': 'raw-data/product_image.jpg'}
                }
            }
        ]
    }
    
    # This would be used for testing with actual S3 data
    # lambda_handler(test_event, None)