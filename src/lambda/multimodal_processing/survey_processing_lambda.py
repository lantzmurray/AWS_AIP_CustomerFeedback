#!/usr/bin/env python3
"""
Lambda function for processing customer survey data with SageMaker integration.

This function processes survey data to generate natural language summaries and
statistical analysis. Enhanced for Phase 2 with integration to data validation
layer, SageMaker job management, improved error handling, and TF-IDF clustering
for advanced theme extraction.
"""

import json
import boto3
import os
import uuid
import time
import logging
import sys
import subprocess
import re
from datetime import datetime, timedelta
from urllib.parse import unquote_plus
from botocore.exceptions import ClientError

# For TF-IDF clustering
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import unified quality score calculator
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.quality_score_calculator import calculate_unified_quality_score, log_quality_metrics

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Log sklearn availability
if not SKLEARN_AVAILABLE:
    logger.warning("scikit-learn not available, falling back to keyword-based theme extraction")

# Initialize AWS clients outside handler for reuse
s3_client = boto3.client('s3')
sagemaker = boto3.client('sagemaker')
cloudwatch = boto3.client('cloudwatch')
sqs = boto3.client('sqs')
iam = boto3.client('iam')

# Environment variables for Phase 2 processing
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')
PROCESSED_BUCKET = os.environ.get('PROCESSED_BUCKET', 'lm-ai-feedback-dev')
QUALITY_THRESHOLD = float(os.environ.get('QUALITY_THRESHOLD', '0.7'))
DLQ_URL = os.environ.get('DLQ_URL', '')
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', '3'))
SAGEMAKER_ROLE_ARN = os.environ.get('SAGEMAKER_ROLE_ARN', '')
SAGEMAKER_IMAGE_URI = os.environ.get('SAGEMAKER_IMAGE_URI', '')

def lambda_handler(event, context):
    """
    Lambda handler for survey processing with enhanced validation integration.
    
    Args:
        event (dict): S3 event trigger
        context (dict): Lambda context
        
    Returns:
        dict: Response with processing results
    """
    
    # Track execution time for performance monitoring
    start_time = datetime.now()
    request_id = context.aws_request_id
    
    logger.info(f"Starting survey processing - Request ID: {request_id}")
    
    try:
        # Get S3 object
        record = event['Records'][0]['s3']
        bucket = record['bucket']['name']
        key = unquote_plus(record['object']['key'])
        
        logger.info(f"Processing S3 object: {bucket}/{key} - Request ID: {request_id}")
        
        # Process validated data from Phase 1 (expected format: processed/surveys/filename_validated.json)
        if key.startswith('processed/surveys/') and key.endswith('_validated.json'):
            return process_validated_survey_metadata(bucket, key, request_id)
        
        # Only process survey CSV files for raw processing
        if not key.lower().endswith('.csv'):
            logger.info(f"Skipping non-survey file: {key} - Request ID: {request_id}")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Not a survey file',
                    'request_id': request_id
                })
            }
        
        # Process raw survey file
        return process_raw_survey(bucket, key, request_id, start_time)
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(f"{error_msg} - Request ID: {request_id}")
        send_error_metrics('Survey', request_id)
        
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

def process_validated_survey_metadata(bucket, key, request_id):
    """
    Process validated survey metadata from Phase 1.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        request_id (str): Request ID for tracking
        
    Returns:
        dict: Response with processing results
    """
    
    logger.info(f"Processing validated survey metadata: {key} - Request ID: {request_id}")
    
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
        survey_metadata = validated_content.get('survey_metadata', {})
        survey_key = validated_content.get('survey_key', '')
        
        # Check validation quality score
        validation_score = validation_results.get('quality_score', 0.0)
        if validation_score < QUALITY_THRESHOLD:
            logger.warning(f"Low validation score ({validation_score}) for {key} - Request ID: {request_id}")
        
        # Process actual survey file
        if survey_key:
            processed_result = process_survey_content(bucket, survey_key, request_id)
            
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
                'ValidatedSurvey',
                len(combined_result.get('summaries', [])),
                len(combined_result.get('statistics', {})),
                request_id
            )
            
            logger.info(f"Successfully processed validated survey: {key} - Request ID: {request_id}")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Successfully processed validated survey',
                    'processed_key': processed_key,
                    'request_id': request_id
                })
            }
        else:
            raise ValueError("No survey key found in validated metadata")
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"S3 ClientError processing {key}: {error_code} - {str(e)} - Request ID: {request_id}")
        send_error_metrics('Survey', request_id)
        
        # Send to DLQ if configured
        if DLQ_URL:
            send_to_dlq({'bucket': bucket, 'key': key}, f"S3 ClientError: {error_code} - {str(e)}", request_id)
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f"S3 error: {error_code}", 'request_id': request_id})
        }
    except Exception as e:
        logger.error(f"Error processing validated survey metadata {key}: {str(e)} - Request ID: {request_id}")
        send_error_metrics('Survey', request_id)
        
        # Send to DLQ if configured
        if DLQ_URL:
            send_to_dlq({'bucket': bucket, 'key': key}, str(e), request_id)
        
        raise

def process_raw_survey(bucket, key, request_id, start_time):
    """
    Process raw survey file directly.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        request_id (str): Request ID for tracking
        start_time (datetime): Start time for execution tracking
        
    Returns:
        dict: Response with processing results
    """
    
    logger.info(f"Processing raw survey: {key} - Request ID: {request_id}")
    
    try:
        # Process survey content
        processed_result = process_survey_content(bucket, key, request_id)
        
        # Add raw processing metadata
        processed_result['processing_metadata'] = {
            'source': 'raw_survey',
            'validation_skipped': True,
            'environment': ENVIRONMENT,
            'request_id': request_id
        }
        
        # Save processed results
        processed_key = key.replace('surveys/', 'processed/surveys/').replace('.csv', '_processed.json')
        
        retry_on_failure(
            s3_client.put_object,
            max_retries=MAX_RETRIES,
            Bucket=PROCESSED_BUCKET,
            Key=processed_key,
            Body=json.dumps(processed_result, default=str),
            ContentType='application/json',
            Metadata={
                'request_id': request_id,
                'source': 'raw_survey',
                'processing_timestamp': datetime.now().isoformat()
            }
        )
        
        # Send success metrics
        send_processing_metrics(
            'RawSurvey',
            len(processed_result.get('summaries', [])),
            len(processed_result.get('statistics', {})),
            request_id
        )
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        send_execution_time_metric(execution_time, True, request_id)
        
        logger.info(f"Successfully processed raw survey: {key} - Request ID: {request_id}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully processed survey',
                'processed_key': processed_key,
                'request_id': request_id
            })
        }
        
    except Exception as e:
        logger.error(f"Error processing raw survey {key}: {str(e)} - Request ID: {request_id}")
        send_error_metrics('Survey', request_id)
        raise

def process_survey_content(bucket, key, request_id):
    """
    Process survey content using SageMaker with optimized memory usage.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key
        request_id (str): Request ID for tracking
        
    Returns:
        dict: Processed survey results
    """
    
    results = {}
    
    # Submit SageMaker processing job
    try:
        job_name = f"survey-processing-{uuid.uuid4()}"
        input_uri = f"s3://{bucket}/{key}"
        output_uri = f"s3://{PROCESSED_BUCKET}/sagemaker-output/{request_id}/"
        
        # Create SageMaker processing job
        job_config = create_sagemaker_job_config(job_name, input_uri, output_uri)
        
        # Submit the job
        sagemaker.create_processing_job(**job_config)
        
        # Wait for job completion (with timeout)
        job_result = wait_for_sagemaker_job(job_name, request_id)
        
        if job_result['status'] == 'Completed':
            # Get processed results from S3
            results_key = f"sagemaker-output/{request_id}/survey_summaries.json"
            results_response = retry_on_failure(
                s3_client.get_object,
                max_retries=MAX_RETRIES,
                Bucket=PROCESSED_BUCKET,
                Key=results_key
            )
            results = json.loads(results_response['Body'].read().decode('utf-8'))
            
            # Get statistics
            stats_key = f"sagemaker-output/{request_id}/survey_statistics.json"
            stats_response = retry_on_failure(
                s3_client.get_object,
                max_retries=MAX_RETRIES,
                Bucket=PROCESSED_BUCKET,
                Key=stats_key
            )
            statistics = json.loads(stats_response['Body'].read().decode('utf-8'))
            
            # Get trend analysis
            trends_key = f"sagemaker-output/{request_id}/trend_analysis.json"
            trends_response = retry_on_failure(
                s3_client.get_object,
                max_retries=MAX_RETRIES,
                Bucket=PROCESSED_BUCKET,
                Key=trends_key
            )
            trend_analysis = json.loads(trends_response['Body'].read().decode('utf-8'))
            
            results['summaries'] = results.get('summaries', [])
            results['statistics'] = statistics
            results['trend_analysis'] = trend_analysis
            results['sagemaker_job_name'] = job_name
            results['processing_status'] = 'COMPLETED'
            
        else:
            results['processing_status'] = 'FAILED'
            results['error'] = job_result.get('error', 'Unknown SageMaker error')
            
    except Exception as e:
        logger.error(f"SageMaker processing failed: {str(e)} - Request ID: {request_id}")
        results['processing_status'] = 'ERROR'
        results['error'] = str(e)
        results['summaries'] = []
        results['statistics'] = {}
        results['trend_analysis'] = {}
    
    # Add enhanced metadata
    results['metadata'] = {
        'survey_key': key,
        'survey_id': os.path.basename(key).split('.')[0],
        'processed_timestamp': datetime.now().isoformat(),
        'environment': ENVIRONMENT,
        'request_id': request_id,
        'processing_version': '2.0'  # Phase 2 version
    }
    
    # Add insights for better analysis
    results['insights'] = extract_survey_insights(results)
    
    return results

def create_sagemaker_job_config(job_name, input_uri, output_uri):
    """
    Create SageMaker processing job configuration.
    
    Args:
        job_name (str): Unique job name
        input_uri (str): S3 input URI
        output_uri (str): S3 output URI
        
    Returns:
        dict: SageMaker job configuration
    """
    
    return {
        'ProcessingJobName': job_name,
        'ProcessingInputs': [
            {
                'InputName': 'input',
                'S3Input': {
                    'S3Uri': input_uri,
                    'LocalPath': '/opt/ml/processing/input'
                }
            },
            {
                'InputName': 'script',
                'CodeInput': {
                    'S3Uri': f"s3://{PROCESSED_BUCKET}/code/survey_processing_script.py",
                    'LocalPath': '/opt/ml/processing/input/code/survey_processing_script.py'
                }
            }
        ],
        'ProcessingOutputConfig': {
            'Outputs': [
                {
                    'OutputName': 'output',
                    'S3Output': {
                        'S3Uri': output_uri,
                        'LocalPath': '/opt/ml/processing/output'
                    }
                }
            ]
        },
        'ProcessingResources': {
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': 'ml.m5.large',
                'VolumeSizeInGB': 30
            }
        },
        'RoleArn': SAGEMAKER_ROLE_ARN,
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 3600  # 1 hour max
        }
    }

def wait_for_sagemaker_job(job_name, request_id):
    """
    Wait for SageMaker job completion with timeout handling.
    
    Args:
        job_name (str): SageMaker job name
        request_id (str): Request ID for tracking
        
    Returns:
        dict: Job result with status and error if any
    """
    
    max_wait_time = 3600  # 1 hour max wait
    start_time = time.time()
    
    while True:
        try:
            response = sagemaker.describe_processing_job(ProcessingJobName=job_name)
            job_status = response['ProcessingJobStatus']
            
            if job_status in ['Completed', 'Failed', 'Stopped']:
                if job_status == 'Completed':
                    return {'status': 'Completed'}
                else:
                    failure_reason = response.get('FailureReason', 'Unknown reason')
                    return {'status': 'Failed', 'error': failure_reason}
            
            if time.time() - start_time > max_wait_time:
                return {'status': 'Timeout', 'error': f'Job timed out after {max_wait_time} seconds'}
            
            time.sleep(30)  # Wait 30 seconds before checking again
            
        except Exception as e:
            logger.warning(f"Error checking SageMaker job status: {str(e)} - Request ID: {request_id}")
            time.sleep(30)  # Wait before retrying

def extract_survey_insights(processed_survey):
    """
    Extract additional insights from processed survey.
    
    Args:
        processed_survey (dict): Processed survey data
        
    Returns:
        dict: Additional insights
    """
    
    insights = {}
    
    # Summary insights
    summaries = processed_survey.get('summaries', [])
    if summaries:
        insights['summary_count'] = len(summaries)
        
        # Extract common themes from summaries
        all_comments = [summary.get('comments', '') for summary in summaries]
        insights['common_themes'] = extract_common_themes(all_comments, request_id)
        
        # Priority analysis
        priority_scores = [summary.get('priority_score', 0) for summary in summaries]
        if priority_scores:
            insights['priority_analysis'] = {
                'high_priority_count': sum(1 for score in priority_scores if score > 5),
                'avg_priority_score': sum(priority_scores) / len(priority_scores),
                'urgent_items': len([s for s in summaries if s.get('priority_score', 0) > 7])
            }
    
    # Statistics insights
    statistics = processed_survey.get('statistics', {})
    if statistics:
        insights['satisfaction_analysis'] = {
            'avg_satisfaction': statistics.get('avg_satisfaction', 0),
            'satisfaction_distribution': statistics.get('satisfaction_distribution', {}),
            'response_rate': statistics.get('response_rate', 0)
        }
        
        insights['demographic_insights'] = statistics.get('demographics', {})
        insights['rating_breakdown'] = statistics.get('rating_breakdown', {})
    
    # Trend insights
    trend_analysis = processed_survey.get('trend_analysis', {})
    if trend_analysis:
        insights['temporal_trends'] = trend_analysis.get('temporal_trends', {})
        insights['top_improvement_areas'] = trend_analysis.get('top_improvement_areas', {})
    
    return insights

def extract_common_themes(comments, request_id=None):
    """
    Extract common themes from survey comments using TF-IDF clustering.
    
    This function implements advanced theme extraction using TF-IDF vectorization and
    K-means clustering to identify meaningful themes in survey comments. It automatically
    determines the optimal number of clusters using silhouette scoring and provides detailed
    theme information including keywords and comment counts.
    
    Args:
        comments (list): List of survey comments
        request_id (str, optional): Request ID for tracking
        
    Returns:
        list: Common themes identified through clustering
        
    Note:
        - Uses TF-IDF vectorization with n-grams (1,2) for better context
        - Automatically determines optimal cluster count using silhouette score
        - Falls back to keyword-based extraction if scikit-learn is unavailable
        - Sends CloudWatch metrics for theme tracking and performance monitoring
    """
    
    # Filter out empty comments
    valid_comments = [comment.strip() for comment in comments if comment and comment.strip()]
    
    if not valid_comments:
        logger.warning("No valid comments provided for theme extraction")
        return []
    
    # Use TF-IDF clustering if scikit-learn is available
    if SKLEARN_AVAILABLE:
        return extract_themes_tfidf_clustering(valid_comments, request_id)
    else:
        # Fallback to keyword-based extraction
        logger.warning("Using fallback keyword-based theme extraction")
        return extract_themes_keyword_based(valid_comments, request_id)

def extract_themes_tfidf_clustering(comments, request_id=None):
    """
    Extract themes using TF-IDF vectorization and K-means clustering.
    
    This function implements advanced machine learning-based theme extraction:
    1. Preprocesses text by removing special characters and normalizing
    2. Creates TF-IDF vectors with configurable parameters
    3. Determines optimal cluster count using silhouette scoring
    4. Applies K-means clustering with optimal parameters
    5. Extracts meaningful theme names from cluster centroids
    
    Args:
        comments (list): List of valid survey comments
        request_id (str, optional): Request ID for tracking
        
    Returns:
        list: Common themes identified through clustering
        
    Technical Details:
        - TF-IDF max_features: 100 (top terms by document frequency)
        - ngram_range: (1,2) for unigrams and bigrams
        - min_df: 2 (ignore terms in <2 documents)
        - max_df: 0.8 (ignore terms in >80% of documents)
        - K-means: 10 initializations, fixed random_state for reproducibility
    """
    
    try:
        # Preprocess comments
        processed_comments = [preprocess_text(comment) for comment in comments]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        
        # Fit and transform the comments
        tfidf_matrix = vectorizer.fit_transform(processed_comments)
        
        # Determine optimal number of clusters using silhouette score
        max_clusters = min(8, len(comments) // 2)  # Cap at 8 clusters
        if max_clusters < 2:
            max_clusters = 2
            
        best_score = -1
        best_k = 2
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            score = silhouette_score(tfidf_matrix, cluster_labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        # Apply K-means with optimal number of clusters
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Extract themes from clusters
        feature_names = vectorizer.get_feature_names_out()
        themes = []
        
        for cluster_id in range(best_k):
            # Get comments in this cluster
            cluster_comments = [comments[i] for i in range(len(comments)) if cluster_labels[i] == cluster_id]
            
            # Get top terms for this cluster
            center = kmeans.cluster_centers_[cluster_id]
            top_indices = center.argsort()[-5:][::-1]  # Get top 5 terms
            top_terms = [feature_names[i] for i in top_indices]
            
            # Create a theme name from top terms
            theme_name = generate_theme_name(top_terms, cluster_comments)
            
            themes.append({
                'theme': theme_name,
                'keywords': top_terms,
                'comment_count': len(cluster_comments),
                'cluster_id': cluster_id
            })
        
        # Sort themes by comment count (most common first)
        themes.sort(key=lambda x: x['comment_count'], reverse=True)
        
        # Send CloudWatch metrics for theme tracking
        send_theme_extraction_metrics(themes, 'tfidf_clustering', request_id)
        
        # Return just the theme names for backward compatibility
        return [theme['theme'] for theme in themes[:5]]
        
    except Exception as e:
        logger.error(f"TF-IDF clustering failed: {str(e)} - Request ID: {request_id}")
        # Fallback to keyword-based extraction
        return extract_themes_keyword_based(comments, request_id)

def extract_themes_keyword_based(comments, request_id=None):
    """
    Enhanced fallback keyword-based theme extraction.
    
    This function provides a robust fallback when scikit-learn is unavailable.
    It uses improved keyword matching with word boundaries and expanded theme
    categories for better coverage.
    
    Args:
        comments (list): List of survey comments
        request_id (str, optional): Request ID for tracking
        
    Returns:
        list: Common themes identified through keyword matching
        
    Enhanced Features:
        - Word boundary matching for accurate keyword detection
        - Expanded theme categories (8 vs original 5)
        - Regex-based pattern matching
        - CloudWatch metrics integration
    """
    
    # Enhanced keyword-based theme extraction
    theme_keywords = {
        'product_quality': ['quality', 'product', 'feature', 'performance', 'design', 'reliable', 'durable'],
        'customer_service': ['service', 'support', 'help', 'staff', 'representative', 'responsive', 'friendly'],
        'price_value': ['price', 'cost', 'value', 'expensive', 'cheap', 'worth', 'affordable', 'budget'],
        'delivery_shipping': ['delivery', 'shipping', 'package', 'arrived', 'fast', 'slow', 'timely', 'delayed'],
        'user_experience': ['experience', 'interface', 'usability', 'easy', 'difficult', 'intuitive', 'confusing'],
        'technical_issues': ['bug', 'error', 'crash', 'glitch', 'problem', 'issue', 'broken'],
        'billing_payment': ['billing', 'payment', 'charge', 'invoice', 'refund', 'transaction'],
        'product_features': ['feature', 'functionality', 'capability', 'option', 'setting', 'customization']
    }
    
    theme_counts = {theme: 0 for theme in theme_keywords.keys()}
    all_text = ' '.join(comments).lower()
    
    # Count keyword occurrences
    for theme, keywords in theme_keywords.items():
        for keyword in keywords:
            # Use word boundaries for more accurate matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = re.findall(pattern, all_text)
            theme_counts[theme] += len(matches)
    
    # Return top themes
    sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    top_themes = [theme for theme, count in sorted_themes[:5] if count > 0]
    
    # Send CloudWatch metrics for theme tracking
    send_theme_extraction_metrics(
        [{'theme': theme, 'comment_count': count, 'keywords': theme_keywords[theme]}
         for theme, count in sorted_themes[:5] if count > 0],
        'keyword_based',
        request_id
    )
    
    return top_themes

def preprocess_text(text):
    """
    Preprocess text for TF-IDF vectorization.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def generate_theme_name(top_terms, cluster_comments):
    """
    Generate a meaningful theme name from top terms and cluster comments.
    
    Args:
        top_terms (list): Top terms from the cluster
        cluster_comments (list): Comments in the cluster
        
    Returns:
        str: Generated theme name
    """
    
    # Try to match with known theme categories
    term_string = ' '.join(top_terms).lower()
    
    # Theme category mapping
    theme_categories = {
        'product_quality': ['quality', 'product', 'feature', 'performance', 'design', 'reliable'],
        'customer_service': ['service', 'support', 'help', 'staff', 'representative', 'responsive'],
        'price_value': ['price', 'cost', 'value', 'expensive', 'cheap', 'worth', 'affordable'],
        'delivery_shipping': ['delivery', 'shipping', 'package', 'arrived', 'fast', 'timely'],
        'user_experience': ['experience', 'interface', 'usability', 'easy', 'intuitive', 'user'],
        'technical_issues': ['bug', 'error', 'crash', 'glitch', 'problem', 'issue', 'broken'],
        'billing_payment': ['billing', 'payment', 'charge', 'invoice', 'refund', 'transaction'],
        'product_features': ['feature', 'functionality', 'capability', 'option', 'setting']
    }
    
    # Find best matching category
    best_match = 'general_feedback'
    best_score = 0
    
    for category, keywords in theme_categories.items():
        score = sum(1 for keyword in keywords if keyword in term_string)
        if score > best_score:
            best_score = score
            best_match = category
    
    # Convert category name to human-readable format
    theme_names = {
        'product_quality': 'Product Quality',
        'customer_service': 'Customer Service',
        'price_value': 'Price & Value',
        'delivery_shipping': 'Delivery & Shipping',
        'user_experience': 'User Experience',
        'technical_issues': 'Technical Issues',
        'billing_payment': 'Billing & Payment',
        'product_features': 'Product Features',
        'general_feedback': 'General Feedback'
    }
    
    return theme_names.get(best_match, 'General Feedback')

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
        'survey_key': validated_data.get('survey_key', processed_result['metadata']['survey_key']),
        'summaries': processed_result.get('summaries', []),
        'statistics': processed_result.get('statistics', {}),
        'trend_analysis': processed_result.get('trend_analysis', {}),
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
    
    # Check if SageMaker processing was successful
    if processed_result.get('processing_status') == 'COMPLETED':
        processing_score += 0.4
    
    # Check if summaries were generated
    if processed_result.get('summaries'):
        processing_score += 0.2
    
    # Check if statistics were calculated
    if processed_result.get('statistics'):
        processing_score += 0.2
    
    # Check if trend analysis was performed
    if processed_result.get('trend_analysis'):
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
            'lambda_function': 'SurveyProcessingFunctionLM'
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

def send_processing_metrics(data_type, summary_count, statistic_count, request_id):
    """
    Send processing metrics to CloudWatch for performance monitoring.
    
    Args:
        data_type (str): Type of data processed
        summary_count (int): Number of summaries generated
        statistic_count (int): Number of statistics calculated
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
                        {'Name': 'Function', 'Value': 'SurveyProcessing'}
                    ]
                },
                {
                    'MetricName': 'SummaryCount',
                    'Value': summary_count,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'DataType', 'Value': data_type},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                },
                {
                    'MetricName': 'StatisticCount',
                    'Value': statistic_count,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'DataType', 'Value': data_type},
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                },
                {
                    'MetricName': 'ProcessingQualityScore',
                    'Value': calculate_processing_quality_score(data_type, summary_count, statistic_count),
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

def calculate_processing_quality_score(data_type, summary_count, statistic_count):
    """
    Calculate processing quality score based on processing results.
    
    Args:
        data_type (str): Type of data processed
        summary_count (int): Number of summaries generated
        statistic_count (int): Number of statistics calculated
        
    Returns:
        float: Processing quality score (0-1)
    """
    
    score = 0.0
    
    # Base score for successful processing
    score += 0.4
    
    # Bonus for summary generation
    if summary_count > 0:
        score += min(0.3, summary_count * 0.01)
    
    # Bonus for statistical analysis
    if statistic_count > 0:
        score += 0.3
    
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
                        {'Name': 'Function', 'Value': 'SurveyProcessing'}
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
                        {'Name': 'Function', 'Value': 'SurveyProcessing'}
                    ]
                }
            ]
        )
    except Exception as e:
        logger.warning(f"Failed to send execution time metrics: {str(e)} - Request ID: {request_id}")

def send_theme_extraction_metrics(themes, method, request_id=None):
    """
    Send theme extraction metrics to CloudWatch for tracking.
    
    Args:
        themes (list): List of themes identified
        method (str): Method used for theme extraction (tfidf_clustering or keyword_based)
        request_id (str, optional): Request ID for tracking
    """
    
    try:
        # Send theme count metric
        cloudwatch.put_metric_data(
            Namespace='CustomerFeedback/ThemeExtraction',
            MetricData=[
                {
                    'MetricName': 'ThemeCount',
                    'Value': len(themes),
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'Method', 'Value': method},
                        {'Name': 'Environment', 'Value': ENVIRONMENT},
                        {'Name': 'Function', 'Value': 'SurveyProcessing'}
                    ]
                },
                {
                    'MetricName': 'ExtractionMethod',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'Method', 'Value': method},
                        {'Name': 'Environment', 'Value': ENVIRONMENT},
                        {'Name': 'SklearnAvailable', 'Value': str(SKLEARN_AVAILABLE)}
                    ]
                }
            ]
        )
        
        # Send individual theme metrics
        for theme in themes:
            if isinstance(theme, dict):
                theme_name = theme.get('theme', 'unknown')
                comment_count = theme.get('comment_count', 0)
                keywords = theme.get('keywords', [])
            else:
                theme_name = theme
                comment_count = 1
                keywords = []
            
            cloudwatch.put_metric_data(
                Namespace='CustomerFeedback/ThemeExtraction',
                MetricData=[
                    {
                        'MetricName': 'ThemeFrequency',
                        'Value': comment_count,
                        'Unit': 'Count',
                        'Dimensions': [
                            {'Name': 'Theme', 'Value': theme_name},
                            {'Name': 'Method', 'Value': method},
                            {'Name': 'Environment', 'Value': ENVIRONMENT}
                        ]
                    },
                    {
                        'MetricName': 'KeywordCount',
                        'Value': len(keywords),
                        'Unit': 'Count',
                        'Dimensions': [
                            {'Name': 'Theme', 'Value': theme_name},
                            {'Name': 'Method', 'Value': method},
                            {'Name': 'Environment', 'Value': ENVIRONMENT}
                        ]
                    }
                ]
            )
        
        logger.info(f"Sent theme extraction metrics for {len(themes)} themes using {method} - Request ID: {request_id}")
        
    except Exception as e:
        logger.warning(f"Failed to send theme extraction metrics: {str(e)} - Request ID: {request_id}")

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
                    'object': {'key': 'surveys/customer_feedback.csv'}
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
                    'object': {'key': 'processed/surveys/customer_feedback_validated.json'}
                }
            }
        ]
    }
    
    # This would be used for testing with actual S3 data
    # lambda_handler(test_event, None)
    # lambda_handler(validated_test_event, None)