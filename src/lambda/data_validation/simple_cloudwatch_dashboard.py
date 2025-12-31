#!/usr/bin/env python3
"""
Simple CloudWatch Dashboard Creation Script for LM Project
"""

import json
import boto3

def create_simple_dashboard():
    """Create a simple CloudWatch dashboard for LM project."""
    
    cloudwatch = boto3.client('cloudwatch')
    
    # Simple dashboard with basic metrics
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
                    "title": "Text Review Quality Score - LM Project"
                }
            }
        ]
    }
    
    try:
        response = cloudwatch.put_dashboard(
            DashboardName='CustomerFeedbackQualityLM',
            DashboardBody=json.dumps(dashboard_body)
        )
        
        if 'DashboardArn' in response:
            print(f"Successfully created dashboard: {response['DashboardArn']}")
        else:
            print(f"Dashboard creation response: {response}")
        
        return response
        
    except Exception as e:
        print(f"Error creating dashboard: {str(e)}")
        return None

if __name__ == "__main__":
    create_simple_dashboard()