#!/usr/bin/env python3
"""
SageMaker Processing Job Runner

This script runs SageMaker Processing jobs for survey data processing
and other batch processing tasks.
"""

import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
import json
import time
from datetime import datetime

def run_survey_processing_job():
    """
    Run a SageMaker Processing job for survey data processing.
    
    Returns:
        str: Processing job name
    """
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Define the processing job
    script_processor = ScriptProcessor(
        command=['python3'],
        image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        sagemaker_session=sagemaker_session
    )
    
    # Run the processing job
    script_processor.run(
        code='survey_processing_script.py',
        inputs=[
            ProcessingInput(
                source='s3://customer-feedback-analysis-<your-initials>/raw-data/surveys.csv',
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='survey_output',
                source='/opt/ml/processing/output',
                destination='s3://customer-feedback-analysis-<your-initials>/processed-data/surveys/'
            )
        ]
    )
    
    job_name = script_processor.latest_job.job_name
    print(f"Started survey processing job: {job_name}")
    
    return job_name

def run_text_analysis_processing_job(input_path, output_path):
    """
    Run a SageMaker Processing job for text analysis.
    
    Args:
        input_path (str): S3 input path
        output_path (str): S3 output path
        
    Returns:
        str: Processing job name
    """
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Define the processing job
    script_processor = ScriptProcessor(
        command=['python3'],
        image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        sagemaker_session=sagemaker_session
    )
    
    # Run the processing job
    script_processor.run(
        code='text_analysis_script.py',
        inputs=[
            ProcessingInput(
                source=input_path,
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='text_analysis_output',
                source='/opt/ml/processing/output',
                destination=output_path
            )
        ],
        arguments=['--input-format', 'json', '--output-format', 'parquet']
    )
    
    job_name = script_processor.latest_job.job_name
    print(f"Started text analysis processing job: {job_name}")
    
    return job_name

def run_model_evaluation_processing_job(model_results_path, output_path):
    """
    Run a SageMaker Processing job for model evaluation.
    
    Args:
        model_results_path (str): S3 path to model evaluation results
        output_path (str): S3 output path for evaluation report
        
    Returns:
        str: Processing job name
    """
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Define the processing job
    script_processor = ScriptProcessor(
        command=['python3'],
        image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        sagemaker_session=sagemaker_session
    )
    
    # Run the processing job
    script_processor.run(
        code='model_evaluation_script.py',
        inputs=[
            ProcessingInput(
                source=model_results_path,
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='evaluation_output',
                source='/opt/ml/processing/output',
                destination=output_path
            )
        ]
    )
    
    job_name = script_processor.latest_job.job_name
    print(f"Started model evaluation processing job: {job_name}")
    
    return job_name

def monitor_processing_job(job_name, max_wait_time=3600):
    """
    Monitor a SageMaker Processing job until completion.
    
    Args:
        job_name (str): Processing job name
        max_wait_time (int): Maximum wait time in seconds
        
    Returns:
        str: Final job status
    """
    
    sagemaker_client = boto3.client('sagemaker')
    
    start_time = time.time()
    
    while True:
        response = sagemaker_client.describe_processing_job(ProcessingJobName=job_name)
        status = response['ProcessingJobStatus']
        
        print(f"Job {job_name} status: {status}")
        
        if status in ['Completed', 'Failed', 'Stopped']:
            return status
        
        if time.time() - start_time > max_wait_time:
            print(f"Job {job_name} timed out after {max_wait_time} seconds")
            return 'Timeout'
        
        time.sleep(30)  # Wait 30 seconds before checking again

def get_processing_job_logs(job_name, log_stream_prefix=None):
    """
    Get logs from a SageMaker Processing job.
    
    Args:
        job_name (str): Processing job name
        log_stream_prefix (str): Optional log stream prefix
        
    Returns:
        list: List of log events
    """
    
    logs_client = boto3.client('logs')
    
    if not log_stream_prefix:
        log_stream_prefix = f"/aws/sagemaker/ProcessingJobs/{job_name}"
    
    try:
        # Get log streams
        log_streams = logs_client.describe_log_streams(
            logGroupName='/aws/sagemaker/ProcessingJobs',
            logStreamNamePrefix=log_stream_prefix
        )
        
        if not log_streams['logStreams']:
            print(f"No log streams found for job {job_name}")
            return []
        
        # Get the latest log stream
        log_stream_name = log_streams['logStreams'][0]['logStreamName']
        
        # Get log events
        log_events = logs_client.get_log_events(
            logGroupName='/aws/sagemaker/ProcessingJobs',
            logStreamName=log_stream_name
        )
        
        return log_events['events']
        
    except Exception as e:
        print(f"Error getting logs for job {job_name}: {str(e)}")
        return []

def create_processing_job_config(job_type, input_path, output_path, **kwargs):
    """
    Create configuration for a processing job.
    
    Args:
        job_type (str): Type of processing job
        input_path (str): S3 input path
        output_path (str): S3 output path
        **kwargs: Additional configuration parameters
        
    Returns:
        dict: Processing job configuration
    """
    
    base_config = {
        'job_type': job_type,
        'input_path': input_path,
        'output_path': output_path,
        'instance_type': kwargs.get('instance_type', 'ml.m5.large'),
        'instance_count': kwargs.get('instance_count', 1),
        'image_uri': kwargs.get('image_uri', '763104351884.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3'),
        'script_name': kwargs.get('script_name', f'{job_type}_script.py'),
        'arguments': kwargs.get('arguments', []),
        'environment': kwargs.get('environment', {}),
        'created_timestamp': datetime.now().isoformat()
    }
    
    return base_config

def schedule_recurring_processing_job(job_config, schedule_expression):
    """
    Schedule a recurring processing job using EventBridge.
    
    Args:
        job_config (dict): Processing job configuration
        schedule_expression (str): Cron expression for scheduling
        
    Returns:
        str: EventBridge rule name
    """
    
    events_client = boto3.client('events')
    
    # Create EventBridge rule
    rule_name = f"{job_config['job_type']}_processing_schedule"
    
    response = events_client.put_rule(
        Name=rule_name,
        ScheduleExpression=schedule_expression,
        State='ENABLED',
        Description=f"Schedule for {job_config['job_type']} processing job"
    )
    
    # Add SageMaker as target
    events_client.put_targets(
        Rule=rule_name,
        Targets=[
            {
                'Id': '1',
                'Arn': 'arn:aws:sagemaker:us-east-1:123456789012:processing-job',  # Replace with actual account
                'RoleArn': 'arn:aws:iam::123456789012:role/EventBridgeSageMakerRole',  # Replace with actual role
                'Input': json.dumps({
                    'job_config': job_config
                })
            }
        ]
    )
    
    print(f"Created recurring schedule {rule_name} with expression: {schedule_expression}")
    return rule_name

def cleanup_processing_jobs(older_than_hours=24):
    """
    Clean up old processing jobs to avoid clutter.
    
    Args:
        older_than_hours (int): Age in hours for job cleanup
    """
    
    sagemaker_client = boto3.client('sagemaker')
    
    # List processing jobs
    response = sagemaker_client.list_processing_jobs(
        CreationTimeBefore=datetime.fromtimestamp(time.time() - older_than_hours * 3600),
        StatusEquals='Completed'
    )
    
    cleaned_count = 0
    
    for job in response['ProcessingJobSummaries']:
        job_name = job['ProcessingJobName']
        
        try:
            # Delete the processing job
            sagemaker_client.delete_processing_job(ProcessingJobName=job_name)
            print(f"Deleted processing job: {job_name}")
            cleaned_count += 1
        except Exception as e:
            print(f"Error deleting job {job_name}: {str(e)}")
    
    print(f"Cleaned up {cleaned_count} processing jobs older than {older_than_hours} hours")

def estimate_processing_cost(job_config, estimated_duration_minutes=30):
    """
    Estimate the cost of a processing job.
    
    Args:
        job_config (dict): Processing job configuration
        estimated_duration_minutes (int): Estimated job duration in minutes
        
    Returns:
        dict: Cost estimation
    """
    
    # Instance pricing (example for us-east-1)
    instance_pricing = {
        'ml.m5.large': 0.104,  # per hour
        'ml.m5.xlarge': 0.208,
        'ml.m5.2xlarge': 0.416,
        'ml.c5.large': 0.085,
        'ml.c5.xlarge': 0.170,
        'ml.r5.large': 0.126,
        'ml.r5.xlarge': 0.252
    }
    
    instance_type = job_config['instance_type']
    instance_count = job_config['instance_count']
    
    hourly_rate = instance_pricing.get(instance_type, 0.208)  # Default to m5.xlarge pricing
    hourly_cost = hourly_rate * instance_count
    
    duration_hours = estimated_duration_minutes / 60
    estimated_cost = hourly_cost * duration_hours
    
    cost_breakdown = {
        'instance_type': instance_type,
        'instance_count': instance_count,
        'hourly_rate': hourly_rate,
        'estimated_duration_hours': duration_hours,
        'estimated_cost': estimated_cost,
        'currency': 'USD'
    }
    
    return cost_breakdown

if __name__ == "__main__":
    # Example usage
    print("Starting SageMaker Processing Job examples...")
    
    # Run survey processing job
    survey_job_name = run_survey_processing_job()
    
    # Monitor the job
    final_status = monitor_processing_job(survey_job_name)
    print(f"Survey processing job completed with status: {final_status}")
    
    # Get job logs
    logs = get_processing_job_logs(survey_job_name)
    for event in logs[-5:]:  # Show last 5 log events
        print(f"LOG: {event['message']}")
    
    # Estimate cost
    job_config = create_processing_job_config(
        'survey_processing',
        's3://bucket/input',
        's3://bucket/output',
        instance_type='ml.m5.xlarge'
    )
    
    cost_estimate = estimate_processing_cost(job_config, 45)
    print(f"Estimated cost: ${cost_estimate['estimated_cost']:.4f}")
    
    print("SageMaker Processing Job examples completed!")