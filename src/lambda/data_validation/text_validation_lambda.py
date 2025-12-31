#!/usr/bin/env python3
"""
Lambda function for custom text validation of customer feedback.

This function validates text reviews for quality and completeness before
they are processed by the main pipeline.
"""

import json
import boto3
import re
import os
from datetime import datetime

def lambda_handler(event, context):
    """
    Lambda handler for text validation.
    
    Args:
        event (dict): S3 event trigger
        context (dict): Lambda context
        
    Returns:
        dict: Response with validation results
    """
    
    # Get the S3 object
    s3_client = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Only process text reviews
    if not key.endswith('.txt') and not key.endswith('.json'):
        return {
            'statusCode': 200,
            'body': json.dumps('Not a text review file')
        }
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        
        # Parse the content (assuming JSON format)
        if key.endswith('.json'):
            review = json.loads(content)
            text = review.get('review_text', '')
        else:
            text = content
            
        # Validation checks
        validation_results = {
            'file_name': key,
            'timestamp': datetime.now().isoformat(),
            'checks': {
                'min_length': len(text) >= 10,
                'has_product_reference': bool(re.search(r'product|item|purchase', text, re.IGNORECASE)),
                'has_opinion': bool(re.search(r'like|love|hate|good|bad|great|terrible|excellent|poor|recommend', text, re.IGNORECASE)),
                'no_profanity': not bool(re.search(r'badword1|badword2', text, re.IGNORECASE)),  # Add actual profanity list
                'has_structure': text.count('.') >= 1  # At least one sentence
            }
        }
        
        # Calculate overall quality score (simple version)
        passed_checks = sum(1 for check in validation_results['checks'].values() if check)
        total_checks = len(validation_results['checks'])
        validation_results['quality_score'] = passed_checks / total_checks
        
        # Send metrics to CloudWatch
        cloudwatch = boto3.client('cloudwatch')
        cloudwatch.put_metric_data(
            Namespace='CustomerFeedback/TextQuality',
            MetricData=[
                {
                    'MetricName': 'QualityScore',
                    'Value': validation_results['quality_score'],
                    'Unit': 'None',
                    'Dimensions': [
                        {
                            'Name': 'Source',
                            'Value': 'TextReviews'
                        }
                    ]
                }
            ]
        )
        
        # Save validation results
        validation_key = key.replace('raw-data', 'validation-results').replace('.txt', '.json').replace('.json', '_validation.json')
        s3_client.put_object(
            Bucket=bucket,
            Key=validation_key,
            Body=json.dumps(validation_results),
            ContentType='application/json'
        )
        
        print(f"Successfully validated {key}, quality score: {validation_results['quality_score']}")
        
        return {
            'statusCode': 200,
            'body': json.dumps(validation_results)
        }
        
    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error: {str(e)}")
        }

def validate_text_content(text, custom_rules=None):
    """
    Validate text content against a set of rules.
    
    Args:
        text (str): Text content to validate
        custom_rules (dict): Optional custom validation rules
        
    Returns:
        dict: Validation results
    """
    
    default_rules = {
        'min_length': 10,
        'max_length': 5000,
        'require_product_reference': True,
        'require_opinion': True,
        'require_structure': True,
        'profanity_list': ['badword1', 'badword2']  # Add actual profanity words
    }
    
    if custom_rules:
        default_rules.update(custom_rules)
    
    results = {
        'checks': {},
        'passed': True
    }
    
    # Check minimum length
    results['checks']['min_length'] = len(text) >= default_rules['min_length']
    
    # Check maximum length
    results['checks']['max_length'] = len(text) <= default_rules['max_length']
    
    # Check for product reference
    if default_rules['require_product_reference']:
        results['checks']['has_product_reference'] = bool(
            re.search(r'product|item|purchase|buy|order', text, re.IGNORECASE)
        )
    
    # Check for opinion
    if default_rules['require_opinion']:
        results['checks']['has_opinion'] = bool(
            re.search(r'like|love|hate|good|bad|great|terrible|excellent|poor|recommend', text, re.IGNORECASE)
        )
    
    # Check for structure
    if default_rules['require_structure']:
        results['checks']['has_structure'] = text.count('.') >= 1 or text.count('!') >= 1
    
    # Check for profanity
    profanity_pattern = '|'.join(default_rules['profanity_list'])
    results['checks']['no_profanity'] = not bool(re.search(profanity_pattern, text, re.IGNORECASE))
    
    # Overall pass status
    results['passed'] = all(results['checks'].values())
    
    return results

def send_quality_metrics(quality_score, source='TextReviews'):
    """
    Send quality metrics to CloudWatch.
    
    Args:
        quality_score (float): Quality score (0-1)
        source (str): Source of the data
    """
    
    cloudwatch = boto3.client('cloudwatch')
    
    cloudwatch.put_metric_data(
        Namespace='CustomerFeedback/TextQuality',
        MetricData=[
            {
                'MetricName': 'QualityScore',
                'Value': quality_score,
                'Unit': 'None',
                'Dimensions': [
                    {
                        'Name': 'Source',
                        'Value': source
                    }
                ]
            },
            {
                'MetricName': 'ValidationCount',
                'Value': 1,
                'Unit': 'Count',
                'Dimensions': [
                    {
                        'Name': 'Source',
                        'Value': source
                    }
                ]
            }
        ]
    )

if __name__ == "__main__":
    # For local testing
    test_event = {
        'Records': [
            {
                's3': {
                    'bucket': {'name': 'test-bucket'},
                    'object': {'key': 'test-review.json'}
                }
            }
        ]
    }
    
    # This would be used for testing with actual S3 data
    # lambda_handler(test_event, None)