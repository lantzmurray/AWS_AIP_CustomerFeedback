#!/usr/bin/env python3
"""
Lambda function for formatting data for Claude in Amazon Bedrock.

This function processes different data types and formats them for
consumption by Claude foundation model. Now integrates with the new
Phase 3 formatting components for enhanced capabilities.
"""

import json
import boto3
import base64
import os
import sys
from datetime import datetime

# Add the current directory to the Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the new Phase 3 formatters
from foundation_model_formatter import FoundationModelFormatter
from text_formatter import TextFormatter
from image_formatter import ImageFormatter
from audio_formatter import AudioFormatter
from survey_formatter import SurveyFormatter
from metadata_enricher import MetadataEnricher
from quality_assurance import QualityAssurance

# Add feedback loop to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'feedback_loop'))
from model_response_collector import ModelResponseCollector

def lambda_handler(event, context):
    """
    Lambda handler for formatting data for Claude.
    
    Args:
        event (dict): S3 event trigger
        context (dict): Lambda context
        
    Returns:
        dict: Response with formatting results
    """
    
    # Get the S3 object
    s3_client = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Only process processed data files
    if not key.endswith('_processed.json'):
        return {
            'statusCode': 200,
            'body': json.dumps('Not a processed data file')
        }
    
    try:
        # Get the processed data
        response = s3_client.get_object(Bucket=bucket, Key=key)
        processed_data = json.loads(response['Body'].read().decode('utf-8'))
        
        # Initialize the new formatters
        fm_formatter = FoundationModelFormatter()
        metadata_enricher = MetadataEnricher()
        quality_assurance = QualityAssurance()
        
        # Initialize feedback loop components
        response_collector = ModelResponseCollector()
        
        # Determine the data type and format accordingly
        if 'transcript' in processed_data:
            # Audio data - use new AudioFormatter
            audio_formatter = AudioFormatter()
            enriched_data = metadata_enricher.enrich_audio_data(processed_data)
            validation_result = quality_assurance.validate_audio_data(enriched_data)
            
            if not validation_result['is_valid']:
                return {
                    'statusCode': 400,
                    'body': json.dumps(f"Validation failed: {validation_result['errors']}")
                }
            
            formatted_data = audio_formatter.format_for_claude(enriched_data, 'conversation')
            
        elif 'extracted_text' in processed_data:
            # Image data - use new ImageFormatter
            image_formatter = ImageFormatter()
            enriched_data = metadata_enricher.enrich_image_data(processed_data)
            validation_result = quality_assurance.validate_image_data(enriched_data)
            
            if not validation_result['is_valid']:
                return {
                    'statusCode': 400,
                    'body': json.dumps(f"Validation failed: {validation_result['errors']}")
                }
            
            formatted_data = image_formatter.format_for_claude(enriched_data, 'multimodal')
            
        elif 'entities' in processed_data:
            # Text review data - use new TextFormatter
            text_formatter = TextFormatter()
            enriched_data = metadata_enricher.enrich_text_data(processed_data)
            validation_result = quality_assurance.validate_text_data(enriched_data)
            
            if not validation_result['is_valid']:
                return {
                    'statusCode': 400,
                    'body': json.dumps(f"Validation failed: {validation_result['errors']}")
                }
            
            formatted_data = text_formatter.format_for_claude(enriched_data, 'conversation')
            
        elif 'summary_text' in processed_data:
            # Survey data - use new SurveyFormatter
            survey_formatter = SurveyFormatter()
            enriched_data = metadata_enricher.enrich_survey_data(processed_data)
            validation_result = quality_assurance.validate_survey_data(enriched_data)
            
            if not validation_result['is_valid']:
                return {
                    'statusCode': 400,
                    'body': json.dumps(f"Validation failed: {validation_result['errors']}")
                }
            
            formatted_data = survey_formatter.format_for_claude(enriched_data, 'structured')
            
        else:
            return {
                'statusCode': 400,
                'body': json.dumps('Unknown data type')
            }
        
        # Send to Claude for analysis using the foundation model formatter
        start_time = datetime.now()
        claude_response = fm_formatter.send_to_claude(formatted_data)
        processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Capture model response for feedback loop
        try:
            input_data_id = key.split('/')[-1].replace('_processed.json', '')
            response_id = response_collector.capture_model_response(
                model_name='claude',
                model_version='v2',
                input_data_id=input_data_id,
                input_data_type=formatted_data.get('data_type', 'unknown'),
                response_content=claude_response.get('content', ''),
                confidence_score=claude_response.get('confidence'),
                token_count=len(claude_response.get('content', '')),
                processing_time_ms=processing_time_ms,
                error_occurred=claude_response.get('error', False),
                error_message=claude_response.get('error_message')
            )
            print(f"Captured model response for feedback loop: {response_id}")
        except Exception as e:
            print(f"Error capturing model response: {str(e)}")
        
        # Save analysis results with enhanced metadata
        claude_response = fm_formatter.send_to_claude(formatted_data)
        
        # Save the analysis results with enhanced metadata
        analysis_key = key.replace('processed-data', 'analysis-results').replace('_processed.json', '_analysis.json')
        analysis_result = {
            'original_file': key,
            'data_type': formatted_data.get('data_type', 'unknown'),
            'model_used': formatted_data.get('model', 'claude'),
            'formatted_request': formatted_data,
            'claude_response': claude_response,
            'quality_metrics': {
                'validation_passed': True,
                'format_version': '3.0',
                'enrichment_applied': True
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'processing_pipeline': {
                'phase': '3',
                'components_used': [
                    'metadata_enricher',
                    'quality_assurance',
                    'data_formatter',
                    'foundation_model_formatter'
                ]
            }
        }
        
        s3_client.put_object(
            Bucket=bucket,
            Key=analysis_key,
            Body=json.dumps(analysis_result, indent=2),
            ContentType='application/json'
        )
        
        # Send enhanced metrics to CloudWatch
        send_formatting_metrics(
            formatted_data.get('data_type', 'unknown'),
            len(claude_response.get('content', '')),
            validation_result.get('quality_score', 0)
        )
        
        print(f"Successfully formatted and analyzed {key}")
        
        return {
            'statusCode': 200,
            'body': json.dumps('Successfully formatted and analyzed data')
        }
        
    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        send_error_metrics('Formatting')
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error: {str(e)}")
        }

def format_text_data(processed_text):
    """
    Format processed text data for Claude.
    
    Args:
        processed_text (dict): Processed text data
        
    Returns:
        dict: Formatted request for Claude
    """
    
    # Create conversation template for text analysis
    prompt = f"""You are analyzing customer feedback to generate actionable business insights.

CUSTOMER REVIEW:
{processed_text['original_text']}

ANALYSIS DATA:
- Entities: {json.dumps(processed_text['entities'], indent=2)}
- Sentiment: {processed_text['sentiment']} (confidence: {processed_text['sentiment_scores']})
- Key Phrases: {json.dumps([phrase['Text'] for phrase in processed_text['key_phrases']], indent=2)}

METADATA:
- Product ID: {processed_text['metadata'].get('product_id', 'N/A')}
- Customer ID: {processed_text['metadata'].get('customer_id', 'N/A')}
- Review Date: {processed_text['metadata'].get('review_date', 'N/A')}

Please provide:
1. A summary of the customer's main points
2. Key insights for business improvement
3. Recommended actions
4. Sentiment analysis explanation
5. Priority level for follow-up (High/Medium/Low)

Format your response as structured JSON with the following keys:
summary, insights, recommendations, sentiment_analysis, priority_level"""

    return {
        'data_type': 'text',
        'prompt': prompt,
        'max_tokens': 2000,
        'temperature': 0.7
    }

def format_image_data(processed_image, bucket, key):
    """
    Format processed image data for Claude.
    
    Args:
        processed_image (dict): Processed image data
        bucket (str): S3 bucket name
        key (str): S3 object key
        
    Returns:
        dict: Formatted request for Claude
    """
    
    # Get the original image for multimodal analysis
    original_key = key.replace('processed-data', 'raw-data').replace('_processed.json', '.jpg')
    
    try:
        image_base64 = get_image_base64(bucket, original_key)
        
        prompt = f"""You are analyzing product images and extracted text to provide customer feedback insights.

EXTRACTED TEXT:
{processed_image['extracted_text']}

DETECTED LABELS:
{json.dumps([label['Name'] for label in processed_image['labels']], indent=2)}

DETECTED TEXT ELEMENTS:
{json.dumps([text['DetectedText'] for text in processed_image['detected_text']], indent=2)}

METADATA:
- Product ID: {processed_image['metadata'].get('product_id', 'N/A')}
- File Type: {processed_image['metadata'].get('file_extension', 'N/A')}

Please analyze both the image and extracted text to provide:
1. Description of what the image shows
2. Key feedback points from extracted text
3. Product quality assessment
4. Customer concerns or issues identified
5. Recommended actions

Format your response as structured JSON with the following keys:
image_description, key_feedback_points, quality_assessment, concerns, recommended_actions"""

        return {
            'data_type': 'image',
            'prompt': prompt,
            'image_base64': image_base64,
            'max_tokens': 2000,
            'temperature': 0.7
        }
        
    except Exception as e:
        print(f"Error getting image for multimodal analysis: {str(e)}")
        # Fallback to text-only analysis
        return format_text_data({
            'original_text': processed_image['extracted_text'],
            'metadata': processed_image['metadata']
        })

def format_audio_data(processed_audio, bucket, key):
    """
    Format processed audio data for Claude.
    
    Args:
        processed_audio (dict): Processed audio data
        
    Returns:
        dict: Formatted request for Claude
    """
    
    prompt = f"""You are analyzing customer service call transcripts to provide business insights.

CALL TRANSCRIPT:
{processed_audio['transcript']}

SPEAKER INFORMATION:
{json.dumps(processed_audio['speakers'], indent=2)}

ANALYSIS DATA:
- Overall Sentiment: {processed_audio['sentiment']} (confidence: {processed_audio['sentiment_scores']})
- Key Phrases: {json.dumps([phrase['Text'] for phrase in processed_audio['key_phrases']], indent=2)}
- Entities: {json.dumps([entity['Text'] for entity in processed_audio['entities']], indent=2)}

METADATA:
- Call ID: {processed_audio['metadata'].get('call_id', 'N/A')}
- Duration: {processed_audio['metadata'].get('duration', 'N/A')}
- Language: {processed_audio['metadata'].get('language_code', 'N/A')}

Please provide:
1. Summary of the call conversation
2. Customer satisfaction assessment
3. Key issues or concerns raised
4. Service quality evaluation
5. Improvement recommendations
6. Follow-up actions needed

Format your response as structured JSON with the following keys:
call_summary, satisfaction_assessment, key_issues, service_quality, improvements, follow_up_actions"""

    return {
        'data_type': 'audio',
        'prompt': prompt,
        'max_tokens': 3000,
        'temperature': 0.7
    }

def format_survey_data(processed_survey):
    """
    Format processed survey data for Claude.
    
    Args:
        processed_survey (dict): Processed survey data
        
    Returns:
        dict: Formatted request for Claude
    """
    
    prompt = f"""You are analyzing customer survey responses to generate business insights.

SURVEY SUMMARY:
{processed_survey['summary_text']}

RATINGS:
{json.dumps(processed_survey['ratings'], indent=2)}

COMMENTS:
{processed_survey.get('comments', 'No additional comments')}

IMPROVEMENT AREAS:
{json.dumps(processed_survey.get('improvement_areas', []), indent=2)}

PRIORITY SCORE: {processed_survey.get('priority_score', 'N/A')}

METADATA:
- Customer ID: {processed_survey.get('customer_id', 'N/A')}
- Survey Date: {processed_survey.get('survey_date', 'N/A')}

Please provide:
1. Overall customer satisfaction assessment
2. Key themes and patterns
3. Specific improvement areas to address
4. Priority level for follow-up
5. Recommended action plan

Format your response as structured JSON with the following keys:
satisfaction_assessment, key_themes, improvement_areas, priority_level, action_plan"""

    return {
        'data_type': 'survey',
        'prompt': prompt,
        'max_tokens': 2000,
        'temperature': 0.7
    }

def send_to_claude(formatted_request):
    """
    Legacy function - now delegates to FoundationModelFormatter.
    Kept for backward compatibility.
    
    Args:
        formatted_request (dict): Formatted request for Claude
        
    Returns:
        dict: Claude's response
    """
    
    fm_formatter = FoundationModelFormatter()
    return fm_formatter.send_to_claude(formatted_request)

def get_image_base64(bucket, key):
    """
    Get image in base64 format for multimodal analysis.
    
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

def send_formatting_metrics(data_type, response_length, quality_score=0):
    """
    Send enhanced formatting metrics to CloudWatch.
    
    Args:
        data_type (str): Type of data formatted
        response_length (int): Length of Claude response
        quality_score (float): Quality score from validation
    """
    
    cloudwatch = boto3.client('cloudwatch')
    
    cloudwatch.put_metric_data(
        Namespace='CustomerFeedback/Formatting',
        MetricData=[
            {
                'MetricName': 'FormattedCount',
                'Value': 1,
                'Unit': 'Count',
                'Dimensions': [
                    {
                        'Name': 'DataType',
                        'Value': data_type
                    },
                    {
                        'Name': 'Phase',
                        'Value': '3'
                    }
                ]
            },
            {
                'MetricName': 'ResponseLength',
                'Value': response_length,
                'Unit': 'Count',
                'Dimensions': [
                    {
                        'Name': 'DataType',
                        'Value': data_type
                    }
                ]
            },
            {
                'MetricName': 'QualityScore',
                'Value': quality_score,
                'Unit': 'None',
                'Dimensions': [
                    {
                        'Name': 'DataType',
                        'Value': data_type
                    }
                ]
            }
        ]
    )

def send_error_metrics(error_type):
    """
    Send error metrics to CloudWatch.
    
    Args:
        error_type (str): Type of error that occurred
    """
    
    cloudwatch = boto3.client('cloudwatch')
    
    cloudwatch.put_metric_data(
        Namespace='CustomerFeedback/Formatting',
        MetricData=[
            {
                'MetricName': 'FormattingErrors',
                'Value': 1,
                'Unit': 'Count',
                'Dimensions': [
                    {
                        'Name': 'ErrorType',
                        'Value': error_type
                    }
                ]
            }
        ]
    )

def create_conversation_template(data_type, data):
    """
    Create a conversation template for different data types.
    
    Args:
        data_type (str): Type of data
        data (dict): Data to include in template
        
    Returns:
        list: Conversation messages
    """
    
    system_message = {
        "role": "user",
        "content": "You are an expert at analyzing customer feedback and providing actionable business insights."
    }
    
    if data_type == 'text':
        user_message = {
            "role": "user", 
            "content": f"Analyze this customer review: {data.get('text', '')}"
        }
    elif data_type == 'image':
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this product image and provide feedback insights:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data.get('image', '')}}
            ]
        }
    elif data_type == 'audio':
        user_message = {
            "role": "user",
            "content": f"Analyze this customer service call transcript: {data.get('transcript', '')}"
        }
    elif data_type == 'survey':
        user_message = {
            "role": "user",
            "content": f"Analyze this survey response: {data.get('summary', '')}"
        }
    else:
        user_message = {
            "role": "user",
            "content": "Analyze this customer feedback data."
        }
    
    return [system_message, user_message]

if __name__ == "__main__":
    # For local testing
    test_event = {
        'Records': [
            {
                's3': {
                    'bucket': {'name': 'test-bucket'},
                    'object': {'key': 'processed-data/test-review_processed.json'}
                }
            }
        ]
    }
    
    # This would be used for testing with actual S3 data
    # lambda_handler(test_event, None)