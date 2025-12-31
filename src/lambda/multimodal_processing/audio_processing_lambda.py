#!/usr/bin/env python3
"""
Lambda function for processing customer service call recordings with Amazon Transcribe.

This function processes audio recordings to transcribe speech and analyze
sentiment using Amazon Transcribe and Amazon Comprehend. Enhanced for Phase 2
with integration to data validation layer, extended timeout configuration, and improved error handling.
"""

import json
import boto3
import os
import uuid
import time
import logging
import sys
from datetime import datetime, timedelta
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
transcribe = boto3.client('transcribe')
comprehend = boto3.client('comprehend')
cloudwatch = boto3.client('cloudwatch')
sqs = boto3.client('sqs')

# Environment variables for Phase 2 processing
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')
PROCESSED_BUCKET = os.environ.get('PROCESSED_BUCKET', 'lm-ai-feedback-dev')
QUALITY_THRESHOLD = float(os.environ.get('QUALITY_THRESHOLD', '0.7'))
DLQ_URL = os.environ.get('DLQ_URL', '')
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', '3'))
MAX_TRANSCRIPTION_WAIT_TIME = int(os.environ.get('MAX_TRANSCRIPTION_WAIT_TIME', '900'))  # 15 minutes

def lambda_handler(event, context):
    """
    Lambda handler for audio processing with enhanced validation integration.
    
    Args:
        event (dict): S3 event trigger
        context (dict): Lambda context
        
    Returns:
        dict: Response with processing results
    """
    
    # Track execution time for performance monitoring
    start_time = datetime.now()
    request_id = context.aws_request_id
    
    logger.info(f"Starting audio processing - Request ID: {request_id}")
    
    try:
        # Get S3 object
        record = event['Records'][0]['s3']
        bucket = record['bucket']['name']
        key = unquote_plus(record['object']['key'])
        
        logger.info(f"Processing S3 object: {bucket}/{key} - Request ID: {request_id}")
        
        # Process validated data from Phase 1 (expected format: processed/audio/filename_validated.json)
        if key.startswith('processed/audio/') and key.endswith('_validated.json'):
            return process_validated_audio_metadata(bucket, key, request_id)
        
        # Only process audio files for raw processing
        if not key.lower().endswith(('.mp3', '.wav', '.flac')):
            logger.info(f"Skipping non-audio file: {key} - Request ID: {request_id}")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Not an audio file',
                    'request_id': request_id
                })
            }
        
        # Process raw audio file
        return process_raw_audio(bucket, key, request_id, start_time)
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(f"{error_msg} - Request ID: {request_id}")
        send_error_metrics('Audio', request_id)
        
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

def process_validated_audio_metadata(bucket, key, request_id):
    """
    Process validated audio metadata from Phase 1.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        request_id (str): Request ID for tracking
        
    Returns:
        dict: Response with processing results
    """
    
    logger.info(f"Processing validated audio metadata: {key} - Request ID: {request_id}")
    
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
        audio_metadata = validated_content.get('audio_metadata', {})
        audio_key = validated_content.get('audio_key', '')
        
        # Check validation quality score
        validation_score = validation_results.get('quality_score', 0.0)
        if validation_score < QUALITY_THRESHOLD:
            logger.warning(f"Low validation score ({validation_score}) for {key} - Request ID: {request_id}")
        
        # Process actual audio file
        if audio_key:
            processed_result = process_audio_content(bucket, audio_key, request_id)
            
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
                'ValidatedAudio',
                len(combined_result.get('transcript', '')),
                len(combined_result.get('key_phrases', [])),
                len(combined_result.get('entities', [])),
                request_id
            )
            
            logger.info(f"Successfully processed validated audio: {key} - Request ID: {request_id}")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Successfully processed validated audio',
                    'processed_key': processed_key,
                    'request_id': request_id
                })
            }
        else:
            raise ValueError("No audio key found in validated metadata")
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"S3 ClientError processing {key}: {error_code} - {str(e)} - Request ID: {request_id}")
        send_error_metrics('Audio', request_id)
        
        # Send to DLQ if configured
        if DLQ_URL:
            send_to_dlq({'bucket': bucket, 'key': key}, f"S3 ClientError: {error_code} - {str(e)}", request_id)
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f"S3 error: {error_code}", 'request_id': request_id})
        }
    except Exception as e:
        logger.error(f"Error processing validated audio metadata {key}: {str(e)} - Request ID: {request_id}")
        send_error_metrics('Audio', request_id)
        
        # Send to DLQ if configured
        if DLQ_URL:
            send_to_dlq({'bucket': bucket, 'key': key}, str(e), request_id)
        
        raise

def process_raw_audio(bucket, key, request_id, start_time):
    """
    Process raw audio file directly.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        request_id (str): Request ID for tracking
        start_time (datetime): Start time for execution tracking
        
    Returns:
        dict: Response with processing results
    """
    
    logger.info(f"Processing raw audio: {key} - Request ID: {request_id}")
    
    try:
        # Process audio content
        processed_result = process_audio_content(bucket, key, request_id)
        
        # Add raw processing metadata
        processed_result['processing_metadata'] = {
            'source': 'raw_audio',
            'validation_skipped': True,
            'environment': ENVIRONMENT,
            'request_id': request_id
        }
        
        # Save processed results
        processed_key = key.replace('audio/', 'processed/audio/').replace(os.path.splitext(key)[1], '_processed.json')
        
        retry_on_failure(
            s3_client.put_object,
            max_retries=MAX_RETRIES,
            Bucket=PROCESSED_BUCKET,
            Key=processed_key,
            Body=json.dumps(processed_result, default=str),
            ContentType='application/json',
            Metadata={
                'request_id': request_id,
                'source': 'raw_audio',
                'processing_timestamp': datetime.now().isoformat()
            }
        )
        
        # Send success metrics
        send_processing_metrics(
            'RawAudio',
            len(processed_result.get('transcript', '')),
            len(processed_result.get('key_phrases', [])),
            len(processed_result.get('entities', [])),
            request_id
        )
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        send_execution_time_metric(execution_time, True, request_id)
        
        logger.info(f"Successfully processed raw audio: {key} - Request ID: {request_id}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully processed audio',
                'processed_key': processed_key,
                'request_id': request_id
            })
        }
        
    except Exception as e:
        logger.error(f"Error processing raw audio {key}: {str(e)} - Request ID: {request_id}")
        send_error_metrics('Audio', request_id)
        raise

def process_audio_content(bucket, key, request_id):
    """
    Process audio content using Transcribe and Comprehend with optimized memory usage.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        request_id (str): Request ID for tracking
        
    Returns:
        dict: Processed audio results
    """
    
    results = {}
    
    # Start a transcription job
    transcribe = boto3.client('transcribe')
    job_name = f"transcribe-{uuid.uuid4()}"
    output_key = key.replace('audio/', 'transcriptions/').replace(os.path.splitext(key)[1], '.json')
    output_uri = f"s3://{bucket}/{output_key}"
    
    try:
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={
                'MediaFileUri': f"s3://{bucket}/{key}"
            },
            MediaFormat=os.path.splitext(key)[1][1:],  # Remove dot
            LanguageCode='en-US',
            OutputBucketName=bucket,
            OutputKey=output_key,
            Settings={
                'ShowSpeakerLabels': True,
                'MaxSpeakerLabels': 2  # Assuming customer and agent
            }
        )
        
        # Wait for transcription job to complete
        max_wait_time = MAX_TRANSCRIPTION_WAIT_TIME
        start_time = time.time()
        
        while True:
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            
            if job_status in ['COMPLETED', 'FAILED']:
                break
                
            if time.time() - start_time > max_wait_time:
                # Timeout reached, exit gracefully
                logger.warning(f"Transcription job {job_name} timed out after {max_wait_time} seconds - Request ID: {request_id}")
                results['transcription_status'] = 'TIMEOUT'
                results['transcript'] = ''
                results['transcription_error'] = f'Timeout after {max_wait_time} seconds'
                return results
            
            time.sleep(10)  # Wait 10 seconds before checking again
        
        if job_status == 'COMPLETED':
            # Get transcription file
            response = s3_client.get_object(Bucket=bucket, Key=output_key)
            transcription = json.loads(response['Body'].read().decode('utf-8'))
            
            # Extract transcript text
            transcript = transcription['results']['transcripts'][0]['transcript']
            results['transcript'] = transcript
            results['transcription_status'] = 'COMPLETED'
            
            # Extract speaker information if available
            speaker_labels = transcription['results'].get('speaker_labels', {})
            speaker_segments = speaker_labels.get('segments', [])
            results['speakers'] = speaker_segments
            
        else:
            failure_reason = status['TranscriptionJob'].get('FailureReason', 'Unknown reason')
            logger.error(f"Transcription failed: {failure_reason} - Request ID: {request_id}")
            results['transcription_status'] = 'FAILED'
            results['transcript'] = ''
            results['transcription_error'] = failure_reason
            return results
            
    except Exception as e:
        logger.error(f"Transcription job failed: {str(e)} - Request ID: {request_id}")
        results['transcription_status'] = 'ERROR'
        results['transcript'] = ''
        results['transcription_error'] = str(e)
        return results
    
    # Use Amazon Comprehend for sentiment analysis with retry logic
    try:
        if results.get('transcript'):
            sentiment_response = retry_on_failure(
                comprehend.detect_sentiment,
                max_retries=MAX_RETRIES,
                Text=results['transcript'],
                LanguageCode='en'
            )
            results['sentiment'] = sentiment_response['Sentiment']
            results['sentiment_scores'] = sentiment_response['SentimentScore']
        else:
            results['sentiment'] = 'UNKNOWN'
            results['sentiment_scores'] = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Mixed': 0}
    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {str(e)} - Request ID: {request_id}")
        results['sentiment'] = 'UNKNOWN'
        results['sentiment_scores'] = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Mixed': 0}
    
    # Detect key phrases with retry logic
    try:
        if results.get('transcript'):
            key_phrases_response = retry_on_failure(
                comprehend.detect_key_phrases,
                max_retries=MAX_RETRIES,
                Text=results['transcript'],
                LanguageCode='en'
            )
            results['key_phrases'] = key_phrases_response['KeyPhrases']
        else:
            results['key_phrases'] = []
    except Exception as e:
        logger.warning(f"Key phrase detection failed: {str(e)} - Request ID: {request_id}")
        results['key_phrases'] = []
    
    # Detect entities with retry logic
    try:
        if results.get('transcript'):
            entities_response = retry_on_failure(
                comprehend.detect_entities,
                max_retries=MAX_RETRIES,
                Text=results['transcript'],
                LanguageCode='en'
            )
            results['entities'] = entities_response['Entities']
        else:
            results['entities'] = []
    except Exception as e:
        logger.warning(f"Entity detection failed: {str(e)} - Request ID: {request_id}")
        results['entities'] = []
    
    # Add enhanced metadata
    results['metadata'] = {
        'audio_key': key,
        'call_id': os.path.basename(key).split('.')[0],
        'language_code': 'en-US',
        'transcription_job_name': job_name,
        'processed_timestamp': datetime.now().isoformat(),
        'environment': ENVIRONMENT,
        'request_id': request_id,
        'processing_version': '2.0'  # Phase 2 version
    }
    
    # Add insights for better analysis
    results['insights'] = extract_audio_insights(results)
    
    return results

def extract_audio_insights(processed_audio):
    """
    Extract additional insights from processed audio.
    
    Args:
        processed_audio (dict): Processed audio data
        
    Returns:
        dict: Additional insights
    """
    
    insights = {}
    
    # Sentiment analysis insights
    sentiment_scores = processed_audio.get('sentiment_scores', {})
    if sentiment_scores:
        dominant_sentiment = processed_audio.get('sentiment', 'NEUTRAL')
        confidence = max(sentiment_scores.values()) if sentiment_scores else 0
        
        insights['sentiment_analysis'] = {
            'dominant_sentiment': dominant_sentiment,
            'confidence': round(confidence, 3),
            'is_strong_sentiment': confidence > 0.8,
            'is_mixed_sentiment': sentiment_scores.get('Mixed', 0) > 0.3
        }
    
    # Speaker insights
    speakers = processed_audio.get('speakers', [])
    if speakers:
        speaker_times = {}
        for segment in speakers:
            speaker = segment.get('speaker_label', 'unknown')
            if speaker not in speaker_times:
                speaker_times[speaker] = 0
            speaker_times[speaker] += segment.get('end_time', 0) - segment.get('start_time', 0)
        
        insights['speaker_analysis'] = {
            'num_speakers': len(speaker_times.keys()),
            'speaker_durations': speaker_times
        }
    
    # Key topic insights
    key_phrases = [phrase['Text'] for phrase in processed_audio.get('key_phrases', [])]
    insights['key_topics'] = key_phrases[:10]  # Top 10 key phrases
    
    # Entity insights
    entities = processed_audio.get('entities', [])
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
    
    # Transcription quality insights
    transcript_length = len(processed_audio.get('transcript', ''))
    insights['transcription_quality'] = {
        'has_transcript': transcript_length > 0,
        'transcript_length': transcript_length,
        'word_count': len(processed_audio.get('transcript', '').split()),
        'transcription_status': processed_audio.get('transcription_status', 'UNKNOWN')
    }
    
    return insights

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
        'audio_key': validated_data.get('audio_key', processed_result['metadata']['audio_key']),
        'transcript': processed_result.get('transcript', ''),
        'speakers': processed_result.get('speakers', []),
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
    
    # Check if transcription was successful
    if processed_result.get('transcription_status') == 'COMPLETED':
        processing_score += 0.4
    
    # Check if sentiment was detected successfully
    if processed_result.get('sentiment') != 'UNKNOWN':
        processing_score += 0.2
    
    # Check if entities were extracted
    if processed_result.get('insights', {}).get('entity_summary', {}).get('total_entities', 0) > 0:
        processing_score += 0.2
    
    # Check if key topics were identified
    if processed_result.get('insights', {}).get('key_topics'):
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
            'lambda_function': 'AudioProcessingFunctionLM'
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

def send_processing_metrics(data_type, transcript_length, key_phrase_count, entity_count, request_id):
    """
    Send processing metrics to CloudWatch for performance monitoring.
    
    Args:
        data_type (str): Type of data processed
        transcript_length (int): Length of transcript
        key_phrase_count (int): Number of key phrases detected
        entity_count (int): Number of entities detected
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
                        {'Name': 'Function', 'Value': 'AudioProcessing'}
                    ]
                },
                {
                    'MetricName': 'TranscriptLength',
                    'Value': transcript_length,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'DataType', 'Value': data_type},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                },
                {
                    'MetricName': 'KeyPhraseCount',
                    'Value': key_phrase_count,
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
                    'Value': calculate_processing_quality_score(data_type, transcript_length, key_phrase_count, entity_count),
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

def calculate_processing_quality_score(data_type, transcript_length, key_phrase_count, entity_count):
    """
    Calculate processing quality score based on processing results.
    
    Args:
        data_type (str): Type of data processed
        transcript_length (int): Length of transcript
        key_phrase_count (int): Number of key phrases detected
        entity_count (int): Number of entities detected
        
    Returns:
        float: Processing quality score (0-1)
    """
    
    score = 0.0
    
    # Base score for successful processing
    score += 0.4
    
    # Bonus for transcript length
    if transcript_length > 0:
        score += min(0.2, transcript_length * 0.001)  # Cap at 0.2
    
    # Bonus for key phrase extraction
    if key_phrase_count > 0:
        score += min(0.2, key_phrase_count * 0.05)
    
    # Bonus for entity extraction
    if entity_count > 0:
        score += min(0.2, entity_count * 0.05)
    
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
                        {'Name': 'Function', 'Value': 'AudioProcessing'}
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
                        {'Name': 'Function', 'Value': 'AudioProcessing'}
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

if __name__ == "__main__":
    # For local testing
    test_event = {
        'Records': [
            {
                's3': {
                    'bucket': {'name': 'test-bucket'},
                    'object': {'key': 'audio/call_recording.mp3'}
                }
            }
        ]
    }
    
    # Test validated metadata event
    validated_test_event = {
        'Records': [
            {
                's3': {
                    'bucket': {'name': 'test-bucket'},
                    'object': {'key': 'processed/audio/call_recording_validated.json'}
                }
            }
        ]
    }
    
    # This would be used for testing with actual S3 data
    # lambda_handler(test_event, None)
    # lambda_handler(validated_test_event, None)