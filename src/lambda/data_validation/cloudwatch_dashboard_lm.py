#!/usr/bin/env python3
"""
CloudWatch Dashboard Creation Script for LM Project

This script creates CloudWatch dashboards for monitoring data validation
and processing pipeline specifically for LM project.
"""

import json
import boto3

def create_data_quality_dashboard_lm():
    """Create a CloudWatch dashboard for monitoring data quality metrics for LM project."""
    
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
                        ["CustomerFeedbackLM/TextQuality", "QualityScore", "Source", "TextReviewsLM"]
                    ],
                    "period": 300,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Text Review Quality Score - LM Project",
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
                        ["CustomerFeedbackLM/TextQuality", "PassRate", "Source", "TextReviewsLM"]
                    ],
                    "period": 300,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Text Review Pass Rate - LM Project",
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
                "y": 12,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["CustomerFeedbackLM/TextQuality", "ValidationCount", "Source", "TextReviewsLM"]
                    ],
                    "period": 300,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "Daily Validation Count - LM Project"
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 12,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/Lambda", "Errors", "FunctionName", "TextValidationFunctionLM"],
                        [".", "Invocations", ".", "TextValidationFunctionLM"]
                    ],
                    "period": 300,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "Lambda Function Health - LM Project",
                    "yAxis": {
                        "left": {
                            "min": 0
                        }
                    }
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 18,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["CustomerFeedbackLM/TextQuality", "ErrorCount", "Source", "TextReviewsLM"]
                    ],
                    "period": 300,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "Validation Error Count - LM Project"
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 18,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/Lambda", "Duration", "FunctionName", "TextValidationFunctionLM"]
                    ],
                    "period": 300,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Lambda Function Duration - LM Project"
                }
            }
        ]
    }
    
    try:
        response = cloudwatch.put_dashboard(
            DashboardName='CustomerFeedbackQualityLM',
            DashboardBody=json.dumps(dashboard_body)
        )
        
        # Check if dashboard was created successfully
        if 'DashboardArn' in response:
            print(f"Created dashboard: {response['DashboardArn']}")
        else:
            print(f"Dashboard creation failed: {response}")
        
        return response
        
    except Exception as e:
        print(f"Error creating dashboard: {str(e)}")
        return None

def create_s3_metrics_dashboard_lm():
    """Create a CloudWatch dashboard for monitoring S3 metrics for LM project."""
    
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
                        ["AWS/S3", "NumberOfObjects", "BucketName", "lm-ai-feedback-dev", "StorageType", "AllStorageTypes"]
                    ],
                    "period": 86400,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Total Objects in S3 Bucket - LM Project"
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
                        ["AWS/S3", "BucketSizeBytes", "BucketName", "lm-ai-feedback-dev", "StorageType", "StandardStorage"]
                    ],
                    "period": 86400,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Bucket Size (Bytes) - LM Project"
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
                        ["AWS/S3", "AllRequests", "BucketName", "lm-ai-feedback-dev"]
                    ],
                    "period": 300,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "S3 Request Count - LM Project"
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 12,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/S3", "4xxErrors", "BucketName", "lm-ai-feedback-dev"]
                    ],
                    "period": 300,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "S3 4xx Errors - LM Project"
                }
            }
        ]
    }
    
    try:
        response = cloudwatch.put_dashboard(
            DashboardName='CustomerFeedbackS3LM',
            DashboardBody=json.dumps(dashboard_body)
        )
        
        # Check if dashboard was created successfully
        if 'DashboardArn' in response:
            print(f"Created S3 dashboard: {response['DashboardArn']}")
        else:
            print(f"S3 dashboard creation failed: {response}")
        
        return response
        
    except Exception as e:
        print(f"Error creating S3 dashboard: {str(e)}")
        return None

def create_alarms_lm():
    """Create CloudWatch alarms for monitoring LM project."""
    
    cloudwatch = boto3.client('cloudwatch')
    responses = []
    
    # Alarm for low quality scores
    try:
        response = cloudwatch.put_metric_alarm(
            AlarmName='LowQualityScoreLM',
            AlarmDescription='Alert when quality score drops below 0.7 - LM Project',
            MetricName='QualityScore',
            Namespace='CustomerFeedbackLM/TextQuality',
            Statistic='Average',
            Period=300,
            EvaluationPeriods=2,
            Threshold=0.7,
            ComparisonOperator='LessThanThreshold',
            Dimensions=[
                {
                    'Name': 'Source',
                    'Value': 'TextReviewsLM'
                }
            ],
            TreatMissingData='notBreaching'
        )
        responses.append(response)
    except Exception as e:
        print(f"Error creating quality alarm: {str(e)}")
    
    # Alarm for Lambda errors
    try:
        response = cloudwatch.put_metric_alarm(
            AlarmName='LambdaErrorsLM',
            AlarmDescription='Alert when Lambda function errors exceed threshold - LM Project',
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
                    'Value': 'TextValidationFunctionLM'
                }
            ],
            TreatMissingData='notBreaching'
        )
        responses.append(response)
    except Exception as e:
        print(f"Error creating Lambda error alarm: {str(e)}")
    
    # Alarm for high error rate
    try:
        response = cloudwatch.put_metric_alarm(
            AlarmName='HighErrorRateLM',
            AlarmDescription='Alert when validation error rate exceeds 10% - LM Project',
            MetricName='ErrorCount',
            Namespace='CustomerFeedbackLM/TextQuality',
            Statistic='Sum',
            Period=300,
            EvaluationPeriods=2,
            Threshold=10,
            ComparisonOperator='GreaterThanThreshold',
            Dimensions=[
                {
                    'Name': 'Source',
                    'Value': 'TextReviewsLM'
                }
            ],
            TreatMissingData='notBreaching'
        )
        responses.append(response)
    except Exception as e:
        print(f"Error creating high error rate alarm: {str(e)}")
    
    print(f"Created {len(responses)} alarms for LM project")
    return responses

if __name__ == "__main__":
    try:
        print("Creating data quality dashboard for LM project...")
        create_data_quality_dashboard_lm()
        
        print("\nCreating S3 metrics dashboard for LM project...")
        create_s3_metrics_dashboard_lm()
        
        print("\nCreating CloudWatch alarms for LM project...")
        create_alarms_lm()
        
        print("\nDashboards and alarms created successfully for LM project!")
        
    except Exception as e:
        print(f"Error creating dashboards: {str(e)}")