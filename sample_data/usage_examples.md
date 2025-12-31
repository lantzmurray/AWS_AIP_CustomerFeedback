# Usage Examples for Sample Data

This guide provides step-by-step examples of how to use the sample data with the AWS AI project's Lambda functions and processing pipelines.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setting Up Your Environment](#setting-up-your-environment)
3. [Processing Survey Data](#processing-survey-data)
4. [Processing Text Reviews](#processing-text-reviews)
5. [Processing Image Data](#processing-image-data)
6. [Processing Audio Data](#processing-audio-data)
7. [Multimodal Data Workflows](#multimodal-data-workflows)
8. [S3 Integration Examples](#s3-integration-examples)
9. [Error Handling and Troubleshooting](#error-handling-and-troubleshooting)

## Prerequisites

Before using these examples, ensure you have:

- AWS CLI configured with appropriate permissions
- Python 3.8+ installed
- Required AWS SDK packages (boto3, pandas, etc.)
- Access to the AWS services used in the project (S3, Lambda, Comprehend, etc.)

```bash
pip install boto3 pandas numpy
```

## Setting Up Your Environment

### 1. Configure AWS Credentials

```python
import boto3
import os

# Set your AWS credentials (recommended to use IAM roles)
# os.environ['AWS_ACCESS_KEY_ID'] = 'your-access-key'
# os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-secret-key'
# os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# Initialize AWS clients
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')
comprehend = boto3.client('comprehend')
```

### 2. Create S3 Buckets for Data Storage

```python
def create_s3_buckets():
    """Create S3 buckets for storing sample data and processing results"""
    
    buckets = [
        'customer-feedback-raw-data',
        'customer-feedback-processed-data',
        'customer-feedback-validation-results'
    ]
    
    for bucket in buckets:
        try:
            s3_client.create_bucket(Bucket=bucket)
            print(f"Created bucket: {bucket}")
        except Exception as e:
            print(f"Bucket {bucket} may already exist or error: {e}")

# Uncomment to run
# create_s3_buckets()
```

### 3. Upload Sample Data to S3

```python
def upload_sample_data_to_s3():
    """Upload sample data to S3 buckets"""
    
    # Upload survey data
    s3_client.upload_file(
        'sample_data/surveys/customer_feedback_survey.csv',
        'customer-feedback-raw-data',
        'surveys/customer_feedback_survey.csv'
    )
    
    # Upload text reviews
    import glob
    for file_path in glob.glob('sample_data/text_reviews/review_*.txt'):
        file_name = os.path.basename(file_path)
        s3_client.upload_file(
            file_path,
            'customer-feedback-raw-data',
            f'text_reviews/{file_name}'
        )
    
    # Upload image prompts
    for file_path in glob.glob('sample_data/images/prompt_*.txt'):
        file_name = os.path.basename(file_path)
        s3_client.upload_file(
            file_path,
            'customer-feedback-raw-data',
            f'image_prompts/{file_name}'
        )
    
    # Upload audio transcripts
    for file_path in glob.glob('sample_data/audio/transcript_*.txt'):
        file_name = os.path.basename(file_path)
        s3_client.upload_file(
            file_path,
            'customer-feedback-raw-data',
            f'audio_transcripts/{file_name}'
        )
    
    print("Sample data uploaded to S3")

# Uncomment to run
# upload_sample_data_to_s3()
```

## Processing Survey Data

### 1. Running the Survey Processing Script

```python
def process_survey_data():
    """Process survey data using the SageMaker processing script"""
    
    import subprocess
    import json
    
    # Create input and output directories
    os.makedirs('temp/input', exist_ok=True)
    os.makedirs('temp/output', exist_ok=True)
    
    # Copy survey data to input directory
    subprocess.run([
        'cp', 
        'sample_data/surveys/customer_feedback_survey.csv',
        'temp/input/surveys.csv'
    ])
    
    # Run the survey processing script
    subprocess.run([
        'python', 
        'Code/multimodal_processing/survey_processing_script.py',
        '--input-path', 'temp/input',
        '--output-path', 'temp/output'
    ])
    
    # Load and display results
    with open('temp/output/survey_summaries.json', 'r') as f:
        summaries = json.load(f)
    
    with open('temp/output/survey_statistics.json', 'r') as f:
        statistics = json.load(f)
    
    print(f"Processed {len(summaries)} survey responses")
    print(f"Average satisfaction: {statistics.get('avg_satisfaction', 'N/A')}")
    
    return summaries, statistics

# Example usage
# summaries, stats = process_survey_data()
```

### 2. Analyzing Survey Results

```python
def analyze_survey_results(summaries):
    """Analyze processed survey results"""
    
    import pandas as pd
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(summaries)
    
    # Calculate sentiment distribution
    sentiment_counts = df['sentiment_indicators'].apply(
        lambda x: x.get('sentiment_balance', 0)
    ).value_counts()
    
    print("Sentiment Distribution:")
    print(sentiment_counts)
    
    # Find high-priority customers for follow-up
    high_priority = df[df['priority_score'] > 5.0]
    print(f"\nHigh Priority Customers ({len(high_priority)}):")
    for _, customer in high_priority.iterrows():
        print(f"- {customer['customer_id']}: Priority Score {customer['priority_score']}")
    
    return df

# Example usage
# df = analyze_survey_results(summaries)
```

## Processing Text Reviews

### 1. Preparing Text Data for Processing

```python
def prepare_text_data_for_processing():
    """Prepare text review data for Lambda processing"""
    
    import json
    import glob
    
    # Create validation results directory
    os.makedirs('temp/validation-results', exist_ok=True)
    
    for file_path in glob.glob('sample_data/text_reviews/review_*.txt'):
        # Extract customer ID from filename
        customer_id = os.path.basename(file_path).replace('review_', '').replace('.txt', '')
        
        # Read the review content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create JSON structure for processing
        review_data = {
            'customer_id': customer_id,
            'review_text': content,
            'review_date': '2025-12-01',  # Extract from file content in production
            'product_id': f'PROD-{customer_id[-4:]}',  # Example product ID
            'quality_score': 0.85  # Example quality score
        }
        
        # Create validation results
        validation_results = {
            'customer_id': customer_id,
            'quality_score': 0.85,
            'validation_status': 'passed',
            'validation_timestamp': '2025-12-07T17:00:00Z'
        }
        
        # Save validation results
        validation_file = f"temp/validation-results/{customer_id}_validation.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f)
        
        # Save original review data
        raw_file = f"temp/raw-data/{customer_id}.json"
        os.makedirs('temp/raw-data', exist_ok=True)
        with open(raw_file, 'w') as f:
            json.dump(review_data, f)
    
    print("Text data prepared for processing")

# Example usage
# prepare_text_data_for_processing()
```

### 2. Simulating Text Processing Lambda

```python
def simulate_text_processing(customer_id):
    """Simulate the text processing Lambda function locally"""
    
    import json
    from datetime import datetime
    
    # Load validation results
    with open(f'temp/validation-results/{customer_id}_validation.json', 'r') as f:
        validation_results = json.load(f)
    
    # Load original review
    with open(f'temp/raw-data/{customer_id}.json', 'r') as f:
        review = json.load(f)
    
    text = review.get('review_text', '')
    
    # Simulate Amazon Comprehend processing
    # In production, this would use the actual AWS Comprehend service
    mock_entities = [
        {'Text': 'product', 'Type': 'PRODUCT', 'Score': 0.95},
        {'Text': 'customer service', 'Type': 'ORGANIZATION', 'Score': 0.87},
        {'Text': 'delivery', 'Type': 'EVENT', 'Score': 0.82}
    ]
    
    mock_sentiment = 'POSITIVE' if 'excellent' in text.lower() or 'amazing' in text.lower() else 'NEGATIVE' if 'terrible' in text.lower() or 'disappointed' in text.lower() else 'NEUTRAL'
    
    mock_sentiment_scores = {
        'Positive': 0.8 if mock_sentiment == 'POSITIVE' else 0.1,
        'Negative': 0.8 if mock_sentiment == 'NEGATIVE' else 0.1,
        'Neutral': 0.6 if mock_sentiment == 'NEUTRAL' else 0.2,
        'Mixed': 0.1
    }
    
    mock_key_phrases = [
        {'Text': 'customer service', 'Score': 0.95},
        {'Text': 'product quality', 'Score': 0.92},
        {'Text': 'delivery speed', 'Score': 0.87}
    ]
    
    # Combine results
    processed_review = {
        'original_text': text,
        'entities': mock_entities,
        'sentiment': mock_sentiment,
        'sentiment_scores': mock_sentiment_scores,
        'key_phrases': mock_key_phrases,
        'metadata': {
            'product_id': review.get('product_id', ''),
            'customer_id': review.get('customer_id', ''),
            'review_date': review.get('review_date', ''),
            'quality_score': validation_results['quality_score'],
            'processed_timestamp': datetime.now().isoformat()
        }
    }
    
    # Save processed results
    os.makedirs('temp/processed-data', exist_ok=True)
    processed_file = f'temp/processed-data/{customer_id}_processed.json'
    with open(processed_file, 'w') as f:
        json.dump(processed_review, f, indent=2)
    
    print(f"Processed text review for {customer_id}")
    return processed_review

# Example usage
# result = simulate_text_processing('CUST-00001')
```

## Processing Image Data

### 1. Preparing Image Prompts for Processing

```python
def prepare_image_prompts_for_processing():
    """Prepare image prompt data for processing"""
    
    import json
    import glob
    
    os.makedirs('temp/image-prompts', exist_ok=True)
    
    for file_path in glob.glob('sample_data/images/prompt_*.txt'):
        # Extract customer ID from filename
        customer_id = os.path.basename(file_path).replace('prompt_', '').replace('.txt', '')
        
        # Read the prompt content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create prompt data structure
        prompt_data = {
            'customer_id': customer_id,
            'prompt_text': content,
            'upload_date': '2025-12-01',  # Extract from file content in production
            'image_type': 'customer_upload'
        }
        
        # Save prompt data
        prompt_file = f'temp/image-prompts/{customer_id}_prompt.json'
        with open(prompt_file, 'w') as f:
            json.dump(prompt_data, f, indent=2)
    
    print("Image prompts prepared for processing")

# Example usage
# prepare_image_prompts_for_processing()
```

### 2. Simulating Image Processing

```python
def simulate_image_processing(customer_id):
    """Simulate the image processing Lambda function"""
    
    import json
    from datetime import datetime
    
    # Load prompt data
    with open(f'temp/image-prompts/{customer_id}_prompt.json', 'r') as f:
        prompt_data = json.load(f)
    
    # Simulate Amazon Rekognition processing
    # In production, this would process actual images
    mock_labels = [
        {'Name': 'Product', 'Confidence': 95.2},
        {'Name': 'Electronics', 'Confidence': 87.3},
        {'Name': 'Device', 'Confidence': 82.1}
    ]
    
    mock_extracted_text = "Product Model: XYZ-123\nSerial Number: SN456789"
    
    mock_moderation_labels = []  # No content safety issues
    
    mock_faces_detected = 0  # No faces in product images
    
    # Combine results
    processed_image = {
        'image_key': f'images/{customer_id}_image.jpg',
        'extracted_text': mock_extracted_text,
        'labels': mock_labels,
        'detected_text': [{'Text': 'Product Model: XYZ-123', 'Type': 'LINE'}],
        'moderation_labels': mock_moderation_labels,
        'faces_detected': mock_faces_detected,
        'face_details': [],
        'metadata': {
            'customer_id': customer_id,
            'file_extension': '.jpg',
            'processed_timestamp': datetime.now().isoformat()
        }
    }
    
    # Save processed results
    os.makedirs('temp/processed-images', exist_ok=True)
    processed_file = f'temp/processed-images/{customer_id}_image_processed.json'
    with open(processed_file, 'w') as f:
        json.dump(processed_image, f, indent=2)
    
    print(f"Processed image data for {customer_id}")
    return processed_image

# Example usage
# result = simulate_image_processing('CUST-00001')
```

## Processing Audio Data

### 1. Preparing Audio Transcripts for Processing

```python
def prepare_audio_transcripts_for_processing():
    """Prepare audio transcript data for processing"""
    
    import json
    import glob
    
    os.makedirs('temp/audio-transcripts', exist_ok=True)
    
    for file_path in glob.glob('sample_data/audio/transcript_*.txt'):
        # Extract customer ID from filename
        customer_id = os.path.basename(file_path).replace('transcript_', '').replace('.txt', '')
        
        # Read the transcript content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create transcript data structure
        transcript_data = {
            'customer_id': customer_id,
            'transcript_text': content,
            'recording_date': '2025-12-01',  # Extract from file content in production
            'audio_format': 'mp3',
            'duration': 120  # Example duration in seconds
        }
        
        # Save transcript data
        transcript_file = f'temp/audio-transcripts/{customer_id}_transcript.json'
        with open(transcript_file, 'w') as f:
            json.dump(transcript_data, f, indent=2)
    
    print("Audio transcripts prepared for processing")

# Example usage
# prepare_audio_transcripts_for_processing()
```

### 2. Simulating Audio Processing

```python
def simulate_audio_processing(customer_id):
    """Simulate the audio processing Lambda function"""
    
    import json
    from datetime import datetime
    
    # Load transcript data
    with open(f'temp/audio-transcripts/{customer_id}_transcript.json', 'r') as f:
        transcript_data = json.load(f)
    
    transcript = transcript_data.get('transcript_text', '')
    
    # Simulate Amazon Transcribe and Comprehend processing
    mock_speakers = [
        {
            'speaker_label': 'spk_0',
            'start_time': 0.0,
            'end_time': 60.0
        }
    ]
    
    mock_sentiment = 'POSITIVE' if 'excellent' in transcript.lower() or 'amazing' in transcript.lower() else 'NEGATIVE' if 'terrible' in transcript.lower() or 'disappointed' in transcript.lower() else 'NEUTRAL'
    
    mock_sentiment_scores = {
        'Positive': 0.8 if mock_sentiment == 'POSITIVE' else 0.1,
        'Negative': 0.8 if mock_sentiment == 'NEGATIVE' else 0.1,
        'Neutral': 0.6 if mock_sentiment == 'NEUTRAL' else 0.2,
        'Mixed': 0.1
    }
    
    mock_key_phrases = [
        {'Text': 'customer service', 'Score': 0.95},
        {'Text': 'product quality', 'Score': 0.92}
    ]
    
    mock_entities = [
        {'Text': 'product', 'Type': 'PRODUCT', 'Score': 0.95},
        {'Text': 'customer service', 'Type': 'ORGANIZATION', 'Score': 0.87}
    ]
    
    # Combine results
    processed_audio = {
        'audio_key': f'audio/{customer_id}_recording.mp3',
        'transcript': transcript,
        'speakers': mock_speakers,
        'sentiment': mock_sentiment,
        'sentiment_scores': mock_sentiment_scores,
        'key_phrases': mock_key_phrases,
        'entities': mock_entities,
        'metadata': {
            'customer_id': customer_id,
            'duration': transcript_data.get('duration', 120),
            'language_code': 'en-US',
            'transcription_job_name': f'transcribe-{customer_id}',
            'processed_timestamp': datetime.now().isoformat()
        }
    }
    
    # Save processed results
    os.makedirs('temp/processed-audio', exist_ok=True)
    processed_file = f'temp/processed-audio/{customer_id}_audio_processed.json'
    with open(processed_file, 'w') as f:
        json.dump(processed_audio, f, indent=2)
    
    print(f"Processed audio data for {customer_id}")
    return processed_audio

# Example usage
# result = simulate_audio_processing('CUST-00004')
```

## Multimodal Data Workflows

### 1. Processing Complete Customer Profiles

```python
def process_complete_customer_profile(customer_id):
    """Process all available data types for a single customer"""
    
    results = {}
    
    # Process survey data
    try:
        import pandas as pd
        df = pd.read_csv('sample_data/surveys/customer_feedback_survey.csv')
        customer_survey = df[df['customer_id'] == customer_id].iloc[0].to_dict()
        results['survey'] = customer_survey
    except:
        print(f"No survey data found for {customer_id}")
    
    # Process text review
    try:
        text_result = simulate_text_processing(customer_id)
        results['text_review'] = text_result
    except:
        print(f"No text review found for {customer_id}")
    
    # Process image data
    try:
        image_result = simulate_image_processing(customer_id)
        results['image'] = image_result
    except:
        print(f"No image data found for {customer_id}")
    
    # Process audio data
    try:
        audio_result = simulate_audio_processing(customer_id)
        results['audio'] = audio_result
    except:
        print(f"No audio data found for {customer_id}")
    
    # Create consolidated profile
    consolidated_profile = {
        'customer_id': customer_id,
        'data_types_available': list(results.keys()),
        'overall_sentiment': calculate_overall_sentiment(results),
        'priority_score': calculate_priority_score(results),
        'recommendations': generate_recommendations(results),
        'processing_timestamp': datetime.now().isoformat(),
        'detailed_results': results
    }
    
    # Save consolidated profile
    os.makedirs('temp/customer-profiles', exist_ok=True)
    profile_file = f'temp/customer-profiles/{customer_id}_profile.json'
    with open(profile_file, 'w') as f:
        json.dump(consolidated_profile, f, indent=2)
    
    print(f"Processed complete profile for {customer_id}")
    return consolidated_profile

def calculate_overall_sentiment(results):
    """Calculate overall sentiment across all data types"""
    
    sentiments = []
    
    if 'survey' in results:
        rating = results['survey'].get('overall_rating', 3)
        if rating >= 4:
            sentiments.append('POSITIVE')
        elif rating <= 2:
            sentiments.append('NEGATIVE')
        else:
            sentiments.append('NEUTRAL')
    
    if 'text_review' in results:
        sentiments.append(results['text_review'].get('sentiment', 'NEUTRAL'))
    
    if 'audio' in results:
        sentiments.append(results['audio'].get('sentiment', 'NEUTRAL'))
    
    # Determine most common sentiment
    if sentiments:
        return max(set(sentiments), key=sentiments.count)
    return 'NEUTRAL'

def calculate_priority_score(results):
    """Calculate priority score for customer follow-up"""
    
    score = 0
    
    # Low ratings increase priority
    if 'survey' in results:
        rating = results['survey'].get('overall_rating', 3)
        score += (5 - rating) * 2
    
    # Negative sentiment increases priority
    overall_sentiment = calculate_overall_sentiment(results)
    if overall_sentiment == 'NEGATIVE':
        score += 5
    elif overall_sentiment == 'NEUTRAL':
        score += 2
    
    # Multiple data types indicate engaged customer
    score += len(results) * 0.5
    
    return score

def generate_recommendations(results):
    """Generate recommendations based on customer feedback"""
    
    recommendations = []
    
    if 'survey' in results:
        rating = results['survey'].get('overall_rating', 3)
        if rating <= 2:
            recommendations.append("Immediate follow-up required due to low satisfaction")
    
    if 'text_review' in results:
        entities = results['text_review'].get('entities', [])
        for entity in entities:
            if entity.get('Type') == 'ORGANIZATION' and 'service' in entity.get('Text', '').lower():
                recommendations.append("Review customer service processes")
    
    if 'image' in results:
        labels = results['image'].get('labels', [])
        for label in labels:
            if 'damage' in label.get('Name', '').lower() or 'broken' in label.get('Name', '').lower():
                recommendations.append("Investigate product quality issues")
    
    if not recommendations:
        recommendations.append("No immediate action required")
    
    return recommendations

# Example usage
# profile = process_complete_customer_profile('CUST-00001')
```

### 2. Batch Processing Multiple Customers

```python
def batch_process_customers(customer_ids):
    """Process multiple customers in batch"""
    
    profiles = []
    
    for customer_id in customer_ids:
        try:
            profile = process_complete_customer_profile(customer_id)
            profiles.append(profile)
        except Exception as e:
            print(f"Error processing {customer_id}: {e}")
    
    # Generate summary report
    summary = {
        'total_customers': len(customer_ids),
        'successfully_processed': len(profiles),
        'processing_date': datetime.now().isoformat(),
        'sentiment_distribution': analyze_sentiment_distribution(profiles),
        'high_priority_customers': [p['customer_id'] for p in profiles if p.get('priority_score', 0) > 5],
        'profiles': profiles
    }
    
    # Save batch results
    os.makedirs('temp/batch-results', exist_ok=True)
    summary_file = 'temp/batch-results/batch_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Batch processing completed. Processed {len(profiles)} out of {len(customer_ids)} customers.")
    return summary

def analyze_sentiment_distribution(profiles):
    """Analyze sentiment distribution across processed profiles"""
    
    sentiments = [p.get('overall_sentiment', 'NEUTRAL') for p in profiles]
    distribution = {
        'POSITIVE': sentiments.count('POSITIVE'),
        'NEGATIVE': sentiments.count('NEGATIVE'),
        'NEUTRAL': sentiments.count('NEUTRAL')
    }
    
    return distribution

# Example usage
# customer_ids = ['CUST-00001', 'CUST-00003', 'CUST-00004', 'CUST-00007']
# batch_result = batch_process_customers(customer_ids)
```

## S3 Integration Examples

### 1. Setting Up S3 Event Triggers

```python
def setup_s3_event_triggers():
    """Set up S3 event triggers for Lambda functions"""
    
    import json
    
    # Create Lambda function configurations
    lambda_configs = {
        'text-processing-lambda': {
            'runtime': 'python3.8',
            'handler': 'text_processing_lambda.lambda_handler',
            'role': 'arn:aws:iam::123456789012:role/lambda-execution-role',
            'description': 'Process text reviews with Comprehend',
            'environment': {
                'Variables': {
                    'QUALITY_THRESHOLD': '0.7'
                }
            }
        },
        'image-processing-lambda': {
            'runtime': 'python3.8',
            'handler': 'image_processing_lambda.lambda_handler',
            'role': 'arn:aws:iam::123456789012:role/lambda-execution-role',
            'description': 'Process images with Rekognition'
        },
        'audio-processing-lambda': {
            'runtime': 'python3.8',
            'handler': 'audio_processing_lambda.lambda_handler',
            'role': 'arn:aws:iam::123456789012:role/lambda-execution-role',
            'description': 'Process audio with Transcribe and Comprehend'
        }
    }
    
    # S3 bucket notification configurations
    bucket_notifications = {
        'customer-feedback-raw-data': {
            'LambdaFunctionConfigurations': [
                {
                    'LambdaFunctionArn': 'arn:aws:lambda:us-east-1:123456789012:function:text-processing-lambda',
                    'Events': ['s3:ObjectCreated:*'],
                    'Filter': {
                        'Key': {
                            'FilterRules': [
                                {
                                    'Name': 'prefix',
                                    'Value': 'text_reviews/'
                                }
                            ]
                        }
                    }
                },
                {
                    'LambdaFunctionArn': 'arn:aws:lambda:us-east-1:123456789012:function:image-processing-lambda',
                    'Events': ['s3:ObjectCreated:*'],
                    'Filter': {
                        'Key': {
                            'FilterRules': [
                                {
                                    'Name': 'prefix',
                                    'Value': 'images/'
                                }
                            ]
                        }
                    }
                },
                {
                    'LambdaFunctionArn': 'arn:aws:lambda:us-east-1:123456789012:function:audio-processing-lambda',
                    'Events': ['s3:ObjectCreated:*'],
                    'Filter': {
                        'Key': {
                            'FilterRules': [
                                {
                                    'Name': 'prefix',
                                    'Value': 'audio/'
                                }
                            ]
                        }
                    }
                }
            ]
        }
    }
    
    print("S3 event trigger configurations prepared")
    return lambda_configs, bucket_notifications

# Example usage
# lambda_configs, notifications = setup_s3_event_triggers()
```

### 2. Monitoring Processing with CloudWatch

```python
def setup_cloudwatch_monitoring():
    """Set up CloudWatch monitoring for processing metrics"""
    
    cloudwatch = boto3.client('cloudwatch')
    
    # Create custom metrics dashboard
    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["CustomerFeedback/Processing", "ProcessedCount", "DataType", "Text"],
                        [".", ".", ".", "Image"],
                        [".", ".", ".", "Audio"],
                        [".", ".", ".", "Survey"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "Processing Count by Data Type",
                    "period": 300
                }
            },
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["CustomerFeedback/Processing", "ProcessingErrors", "DataType", "Text"],
                        [".", ".", ".", "Image"],
                        [".", ".", ".", "Audio"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "Processing Errors by Data Type",
                    "period": 300
                }
            },
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["CustomerFeedback/Processing", "TextLength", "DataType", "Text"],
                        ["CustomerFeedback/Processing", "TranscriptLength", "DataType", "Audio"],
                        ["CustomerFeedback/Processing", "EntityCount", "DataType", "Text"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "Content Metrics",
                    "period": 300
                }
            }
        ]
    }
    
    try:
        cloudwatch.put_dashboard(
            DashboardName='CustomerFeedbackProcessing',
            DashboardBody=json.dumps(dashboard_body)
        )
        print("CloudWatch dashboard created successfully")
    except Exception as e:
        print(f"Error creating dashboard: {e}")

# Example usage
# setup_cloudwatch_monitoring()
```

## Error Handling and Troubleshooting

### 1. Common Error Scenarios

```python
def handle_common_errors():
    """Handle common error scenarios in data processing"""
    
    error_handlers = {
        'FileNotFoundError': {
            'description': 'Sample data file not found',
            'solution': 'Verify file paths and ensure sample data is available',
            'code': '''
try:
    with open('sample_data/surveys/customer_feedback_survey.csv', 'r') as f:
        data = f.read()
except FileNotFoundError:
    print("Survey data file not found. Check file path.")
    print("Expected path: sample_data/surveys/customer_feedback_survey.csv")
'''
        },
        'ValidationError': {
            'description': 'Data validation failed',
            'solution': 'Check data quality and format requirements',
            'code': '''
if validation_results['quality_score'] < quality_threshold:
    print(f"Quality score too low: {validation_results['quality_score']}")
    print("Minimum required:", quality_threshold)
    return {
        'statusCode': 200,
        'body': json.dumps('Quality score too low')
    }
'''
        },
        'AWSServiceError': {
            'description': 'AWS service call failed',
            'solution': 'Check AWS credentials and service availability',
            'code': '''
try:
    response = comprehend.detect_sentiment(Text=text, LanguageCode='en')
except Exception as e:
    print(f"AWS Comprehend error: {str(e)}")
    print("Check AWS credentials and service limits")
    raise
'''
        },
        'DataFormatError': {
            'description': 'Unexpected data format',
            'solution': 'Validate data structure before processing',
            'code': '''
def validate_data_structure(data, required_fields):
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    return True
'''
        }
    }
    
    return error_handlers

# Example usage
# handlers = handle_common_errors()
# for error_type, handler in handlers.items():
#     print(f"{error_type}: {handler['description']}")
```

### 2. Debugging Helper Functions

```python
def debug_processing_pipeline(customer_id):
    """Debug processing pipeline for a specific customer"""
    
    print(f"Debugging processing pipeline for {customer_id}")
    print("=" * 50)
    
    # Check survey data
    try:
        import pandas as pd
        df = pd.read_csv('sample_data/surveys/customer_feedback_survey.csv')
        customer_survey = df[df['customer_id'] == customer_id]
        if not customer_survey.empty:
            print(f"✓ Survey data found: Rating {customer_survey.iloc[0]['overall_rating']}")
        else:
            print("✗ No survey data found")
    except Exception as e:
        print(f"✗ Error checking survey data: {e}")
    
    # Check text review
    text_file = f'sample_data/text_reviews/review_{customer_id}.txt'
    if os.path.exists(text_file):
        print(f"✓ Text review found: {text_file}")
    else:
        print("✗ No text review found")
    
    # Check image prompt
    image_file = f'sample_data/images/prompt_{customer_id}.txt'
    if os.path.exists(image_file):
        print(f"✓ Image prompt found: {image_file}")
    else:
        print("✗ No image prompt found")
    
    # Check audio transcript
    audio_file = f'sample_data/audio/transcript_{customer_id}.txt'
    if os.path.exists(audio_file):
        print(f"✓ Audio transcript found: {audio_file}")
    else:
        print("✗ No audio transcript found")
    
    print("=" * 50)

# Example usage
# debug_processing_pipeline('CUST-00001')
```

### 3. Performance Monitoring

```python
def monitor_processing_performance():
    """Monitor processing performance and identify bottlenecks"""
    
    import time
    import psutil
    
    def timing_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            print(f"Function {func.__name__}:")
            print(f"  Execution time: {end_time - start_time:.2f} seconds")
            print(f"  Memory usage: {end_memory - start_memory:.2f} MB")
            
            return result
        return wrapper
    
    # Apply monitoring to key functions
    monitored_process_text = timing_decorator(simulate_text_processing)
    monitored_process_image = timing_decorator(simulate_image_processing)
    monitored_process_audio = timing_decorator(simulate_audio_processing)
    
    return {
        'process_text': monitored_process_text,
        'process_image': monitored_process_image,
        'process_audio': monitored_process_audio
    }

# Example usage
# monitored_functions = monitor_processing_performance()
# monitored_functions['process_text']('CUST-00001')
```

## Advanced Usage Examples

### 1. Custom Sentiment Analysis

```python
def custom_sentiment_analysis(text):
    """Implement custom sentiment analysis logic"""
    
    # Define sentiment keywords
    positive_words = [
        'excellent', 'amazing', 'fantastic', 'outstanding', 'perfect',
        'great', 'wonderful', 'impressed', 'satisfied', 'love'
    ]
    
    negative_words = [
        'terrible', 'awful', 'disappointed', 'poor', 'bad',
        'worst', 'hate', 'unacceptable', 'frustrated', 'broken'
    ]
    
    # Count sentiment words
    positive_count = sum(1 for word in positive_words if word in text.lower())
    negative_count = sum(1 for word in negative_words if word in text.lower())
    
    # Calculate sentiment score
    total_words = len(text.split())
    if total_words == 0:
        return 'NEUTRAL', 0.0
    
    sentiment_score = (positive_count - negative_count) / total_words
    
    # Determine sentiment
    if sentiment_score > 0.05:
        return 'POSITIVE', min(sentiment_score * 10, 1.0)
    elif sentiment_score < -0.05:
        return 'NEGATIVE', min(abs(sentiment_score * 10), 1.0)
    else:
        return 'NEUTRAL', 0.5

# Example usage
# sentiment, confidence = custom_sentiment_analysis("This product is amazing and works perfectly!")
# print(f"Sentiment: {sentiment}, Confidence: {confidence}")
```

### 2. Cross-Modal Data Correlation

```python
def analyze_cross_modal_correlation(customer_id):
    """Analyze correlation between different data modalities"""
    
    correlations = {}
    
    # Load all available data for the customer
    data = {}
    
    # Load survey data
    try:
        import pandas as pd
        df = pd.read_csv('sample_data/surveys/customer_feedback_survey.csv')
        survey_row = df[df['customer_id'] == customer_id].iloc[0]
        data['survey'] = {
            'overall_rating': survey_row['overall_rating'],
            'product_quality_rating': survey_row['product_quality_rating'],
            'customer_service_rating': survey_row['customer_service_rating'],
            'would_recommend': survey_row['would_recommend']
        }
    except:
        pass
    
    # Load text review sentiment
    try:
        with open(f'temp/processed-data/{customer_id}_processed.json', 'r') as f:
            text_data = json.load(f)
            data['text'] = {
                'sentiment': text_data['sentiment'],
                'sentiment_score': text_data['sentiment_scores'][text_data['sentiment'].capitalize()]
            }
    except:
        pass
    
    # Load audio sentiment
    try:
        with open(f'temp/processed-audio/{customer_id}_audio_processed.json', 'r') as f:
            audio_data = json.load(f)
            data['audio'] = {
                'sentiment': audio_data['sentiment'],
                'sentiment_score': audio_data['sentiment_scores'][audio_data['sentiment'].capitalize()]
            }
    except:
        pass
    
    # Analyze correlations
    if 'survey' in data and 'text' in data:
        survey_sentiment = 'POSITIVE' if data['survey']['overall_rating'] >= 4 else 'NEGATIVE' if data['survey']['overall_rating'] <= 2 else 'NEUTRAL'
        text_sentiment = data['text']['sentiment']
        
        correlations['survey_text_sentiment_match'] = survey_sentiment == text_sentiment
        correlations['survey_rating_text_sentiment_alignment'] = (data['survey']['overall_rating'] - 3) * 0.2  # Convert to -0.4 to 0.4 range
    
    if 'text' in data and 'audio' in data:
        correlations['text_audio_sentiment_match'] = data['text']['sentiment'] == data['audio']['sentiment']
        correlations['text_audio_sentiment_score_diff'] = abs(data['text']['sentiment_score'] - data['audio']['sentiment_score'])
    
    return correlations

# Example usage
# correlations = analyze_cross_modal_correlation('CUST-00001')
# print("Cross-modal correlations:", correlations)
```

## Conclusion

These examples provide a comprehensive guide for using the sample data with the AWS AI project's processing pipeline. You can adapt these examples to your specific use cases and integrate them into your workflows.

For more detailed information about the data structure and relationships, refer to the [Sample Data Guide](sample_data_guide.md).