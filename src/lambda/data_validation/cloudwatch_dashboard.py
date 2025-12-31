#!/usr/bin/env python3
"""
CloudWatch Dashboard Creation Script

This script creates CloudWatch dashboards for monitoring the data validation
and processing pipeline.
"""

import json
import boto3
from datetime import datetime, timedelta

def create_data_quality_dashboard():
    """
    Create a CloudWatch dashboard for monitoring data quality metrics.
    
    Returns:
        dict: Response from CloudWatch put_dashboard API call
    """
    
    cloudwatch = boto3.client('cloudwatch')
    
    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["CustomerFeedback/TextQuality", "QualityScore", "Source", "TextReviews"]
                    ],
                    "period": 86400,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Text Review Quality Score",
                    "yAxis": {
                        "left": {
                            "min": 0,
                            "max": 1
                        }
                    }
                }
            },
            {
                "type": "metric",
                "x": 0,
                "y": 6,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["CustomerFeedback/DataQuality", "RulesetPassRate", "Ruleset", "customer_reviews_ruleset"]
                    ],
                    "period": 86400,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Glue Data Quality Pass Rate",
                    "yAxis": {
                        "left": {
                            "min": 0,
                            "max": 1
                        }
                    }
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["CustomerFeedback/TextQuality", "ValidationCount", "Source", "TextReviews"]
                    ],
                    "period": 86400,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "Daily Validation Count"
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 6,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/Lambda", "Errors", "FunctionName", "TextValidationFunction"],
                        [".", "Invocations", ".", "."]
                    ],
                    "period": 86400,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "Lambda Function Health",
                    "yAxis": {
                        "left": {
                            "min": 0
                        }
                    }
                }
            }
        ]
    }
    
    response = cloudwatch.put_dashboard(
        DashboardName='CustomerFeedbackQuality',
        DashboardBody=json.dumps(dashboard_body)
    )
    
    print(f"Created dashboard: {response['DashboardArn']}")
    return response

def create_processing_dashboard():
    """
    Create a CloudWatch dashboard for monitoring data processing metrics.
    
    Returns:
        dict: Response from CloudWatch put_dashboard API call
    """
    
    cloudwatch = boto3.client('cloudwatch')
    
    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["CustomerFeedback/Processing", "ProcessingLatency", "DataType", "Text"],
                        [".", ".", "DataType", "Image"],
                        [".", ".", "DataType", "Audio"],
                        [".", ".", "DataType", "Survey"]
                    ],
                    "period": 86400,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Processing Latency by Data Type"
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["CustomerFeedback/Processing", "ProcessedCount", "DataType", "Text"],
                        [".", ".", "DataType", "Image"],
                        [".", ".", "DataType", "Audio"],
                        [".", ".", "DataType", "Survey"]
                    ],
                    "period": 86400,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "Daily Processed Count by Data Type"
                }
            },
            {
                "type": "metric",
                "x": 0,
                "y": 6,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/Comprehend", "CharactersProcessed", ".", "."],
                        ["AWS/Textract", "PagesProcessed", ".", "."],
                        ["AWS/Transcribe", "MinutesProcessed", ".", "."]
                    ],
                    "period": 86400,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "Service Usage Metrics"
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 6,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/Lambda", "Duration", "FunctionName", "TextProcessingFunction"],
                        [".", ".", "FunctionName", "ImageProcessingFunction"],
                        [".", ".", "FunctionName", "AudioProcessingFunction"]
                    ],
                    "period": 86400,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Lambda Function Duration"
                }
            }
        ]
    }
    
    response = cloudwatch.put_dashboard(
        DashboardName='CustomerFeedbackProcessing',
        DashboardBody=json.dumps(dashboard_body)
    )
    
    print(f"Created dashboard: {response['DashboardArn']}")
    return response

def create_cost_monitoring_dashboard():
    """
    Create a CloudWatch dashboard for monitoring costs.
    
    Returns:
        dict: Response from CloudWatch put_dashboard API call
    """
    
    cloudwatch = boto3.client('cloudwatch')
    
    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 24,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/Billing", "EstimatedCharges", "Currency", "USD"]
                    ],
                    "period": 86400,
                    "stat": "Maximum",
                    "region": "us-east-1",
                    "title": "Estimated Daily Charges",
                    "yAxis": {
                        "left": {
                            "min": 0
                        }
                    }
                }
            },
            {
                "type": "metric",
                "x": 0,
                "y": 6,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/Billing", "EstimatedCharges", "Currency", "USD", "Service", "AWS Glue"],
                        [".", ".", ".", ".", "Service", "AWS Lambda"],
                        [".", ".", ".", ".", "Service", "Amazon Comprehend"],
                        [".", ".", ".", ".", "Service", "Amazon Textract"]
                    ],
                    "period": 86400,
                    "stat": "Maximum",
                    "region": "us-east-1",
                    "title": "Service-wise Costs"
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 6,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/Billing", "EstimatedCharges", "Currency", "USD", "Service", "Amazon Transcribe"],
                        [".", ".", ".", ".", "Service", "Amazon SageMaker"],
                        [".", ".", ".", ".", "Service", "Amazon Bedrock"],
                        [".", ".", ".", ".", "Service", "Amazon S3"]
                    ],
                    "period": 86400,
                    "stat": "Maximum",
                    "region": "us-east-1",
                    "title": "Additional Service Costs"
                }
            }
        ]
    }
    
    response = cloudwatch.put_dashboard(
        DashboardName='CustomerFeedbackCosts',
        DashboardBody=json.dumps(dashboard_body)
    )
    
    print(f"Created dashboard: {response['DashboardArn']}")
    return response

def create_custom_metric(namespace, metric_name, value, dimensions=None):
    """
    Publish a custom metric to CloudWatch.
    
    Args:
        namespace (str): CloudWatch namespace
        metric_name (str): Name of the metric
        value (float): Metric value
        dimensions (list): List of dimensions for the metric
    """
    
    cloudwatch = boto3.client('cloudwatch')
    
    metric_data = {
        'MetricName': metric_name,
        'Value': value,
        'Unit': 'None'
    }
    
    if dimensions:
        metric_data['Dimensions'] = dimensions
    
    cloudwatch.put_metric_data(
        Namespace=namespace,
        MetricData=[metric_data]
    )
    
    print(f"Published metric {metric_name} with value {value} to {namespace}")

def create_alarms():
    """
    Create CloudWatch alarms for monitoring.
    
    Returns:
        list: List of alarm creation responses
    """
    
    cloudwatch = boto3.client('cloudwatch')
    responses = []
    
    # Alarm for low quality scores
    response = cloudwatch.put_metric_alarm(
        AlarmName='LowQualityScore',
        AlarmDescription='Alert when quality score drops below 0.7',
        MetricName='QualityScore',
        Namespace='CustomerFeedback/TextQuality',
        Statistic='Average',
        Period=300,
        EvaluationPeriods=2,
        Threshold=0.7,
        ComparisonOperator='LessThanThreshold',
        TreatMissingData='notBreaching'
    )
    responses.append(response)
    
    # Alarm for Lambda errors
    response = cloudwatch.put_metric_alarm(
        AlarmName='LambdaErrors',
        AlarmDescription='Alert when Lambda function errors exceed threshold',
        MetricName='Errors',
        Namespace='AWS/Lambda',
        Statistic='Sum',
        Period=300,
        EvaluationPeriods=2,
        Threshold=5,
        ComparisonOperator='GreaterThanThreshold',
        Dimensions=[
            {
                'Name': 'FunctionName',
                'Value': 'TextValidationFunction'
            }
        ],
        TreatMissingData='notBreaching'
    )
    responses.append(response)
    
    print("Created CloudWatch alarms")
    return responses

if __name__ == "__main__":
    try:
        print("Creating data quality dashboard...")
        create_data_quality_dashboard()
        
        print("\nCreating processing dashboard...")
        create_processing_dashboard()
        
        print("\nCreating cost monitoring dashboard...")
        create_cost_monitoring_dashboard()
        
        print("\nCreating CloudWatch alarms...")
        create_alarms()
        
        print("\nDashboards and alarms created successfully!")
        
    except Exception as e:
        print(f"Error creating dashboards: {str(e)}")