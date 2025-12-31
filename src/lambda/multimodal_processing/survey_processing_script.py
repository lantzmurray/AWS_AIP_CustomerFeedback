#!/usr/bin/env python3
"""
SageMaker Processing script for survey data.

This script processes structured survey data to generate natural language
summaries and statistical analysis. Enhanced for Phase 2 with integration
to data validation layer, improved error handling, and enhanced logging.
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
import logging
import sys
import traceback
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
import boto3

# Configure logging for SageMaker processing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Initialize AWS clients for SageMaker processing
s3_client = boto3.client('s3')
cloudwatch = boto3.client('cloudwatch')

# Environment variables for Phase 2 processing
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'dev')
PROCESSED_BUCKET = os.environ.get('PROCESSED_BUCKET', 'lm-ai-feedback-dev')
QUALITY_THRESHOLD = float(os.environ.get('QUALITY_THRESHOLD', '0.7'))

def process_survey_data(input_path, output_path):
    """
    Process survey data from CSV to natural language summaries.
    
    Args:
        input_path (str): Path to input data
        output_path (str): Path to save processed data
    """
    
    # Read the survey data
    df = pd.read_csv(f"{input_path}/surveys.csv")
    
    print(f"Loaded {len(df)} survey responses")
    
    # Basic data cleaning
    df = df.dropna(subset=['customer_id', 'survey_date'])  # Drop rows with missing key fields
    
    # Convert categorical ratings to numerical
    rating_map = {
        'Very Dissatisfied': 1, 
        'Dissatisfied': 2, 
        'Neutral': 3, 
        'Satisfied': 4, 
        'Very Satisfied': 5
    }
    
    # Apply rating conversion to relevant columns
    for col in df.columns:
        if 'rating' in col.lower() or 'satisfaction' in col.lower():
            df[col] = df[col].map(rating_map).fillna(df[col])
    
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(df)
    
    # Generate natural language summaries for each survey
    summaries = generate_survey_summaries(df)
    
    # Generate trend analysis
    trend_analysis = generate_trend_analysis(df)
    
    # Save the processed data
    with open(f"{output_path}/survey_summaries.json", 'w') as f:
        json.dump(summaries, f, indent=2)
    
    with open(f"{output_path}/survey_statistics.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    with open(f"{output_path}/trend_analysis.json", 'w') as f:
        json.dump(trend_analysis, f, indent=2)
    
    print(f"Processed {len(summaries)} survey responses")
    print(f"Saved results to {output_path}")

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for survey data.
    
    Args:
        df (pd.DataFrame): Survey data
        
    Returns:
        dict: Summary statistics
    """
    
    stats = {
        'total_surveys': len(df),
        'response_rate': calculate_response_rate(df),
        'avg_satisfaction': df['overall_satisfaction'].mean() if 'overall_satisfaction' in df.columns else None,
        'satisfaction_distribution': df['overall_satisfaction'].value_counts().to_dict() if 'overall_satisfaction' in df.columns else {},
        'top_issues': df['improvement_area'].value_counts().head(5).to_dict() if 'improvement_area' in df.columns else {},
        'demographics': analyze_demographics(df),
        'rating_breakdown': analyze_rating_breakdown(df),
        'temporal_trends': analyze_temporal_trends(df)
    }
    
    return stats

def generate_survey_summaries(df):
    """
    Generate natural language summaries for each survey response.
    
    Args:
        df (pd.DataFrame): Survey data
        
    Returns:
        list: List of survey summaries
    """
    
    summaries = []
    
    for _, row in df.iterrows():
        summary = {
            'customer_id': row['customer_id'],
            'survey_date': row['survey_date'],
            'summary_text': generate_summary(row),
            'ratings': extract_ratings(row),
            'comments': row.get('comments', ''),
            'improvement_areas': extract_improvement_areas(row),
            'sentiment_indicators': extract_sentiment_indicators(row),
            'priority_score': calculate_priority_score(row)
        }
        summaries.append(summary)
    
    return summaries

def generate_summary(row):
    """
    Generate a natural language summary of a survey response.
    
    Args:
        row (pd.Series): Survey response row
        
    Returns:
        str: Natural language summary
    """
    
    summary_parts = []
    
    # Customer identification
    customer_id = row.get('customer_id', 'Unknown')
    summary_parts.append(f"Customer {customer_id}")
    
    # Overall satisfaction
    if 'overall_satisfaction' in row and pd.notna(row['overall_satisfaction']):
        satisfaction_level = get_satisfaction_level(row['overall_satisfaction'])
        summary_parts.append(f"reported being {satisfaction_level}")
    
    # Product/service ratings
    product_rating = row.get('product_rating')
    service_rating = row.get('service_rating')
    
    if pd.notna(product_rating):
        summary_parts.append(f"rated the product {product_rating}/5")
    
    if pd.notna(service_rating):
        summary_parts.append(f"rated customer service {service_rating}/5")
    
    # Improvement areas
    if 'improvement_area' in row and pd.notna(row['improvement_area']):
        improvement_area = row['improvement_area']
        summary_parts.append(f"suggested improvements in {improvement_area}")
    
    # Comments
    if 'comments' in row and pd.notna(row['comments']) and len(str(row['comments']).strip()) > 0:
        comments = str(row['comments']).strip()
        if len(comments) > 100:
            comments = comments[:100] + "..."
        summary_parts.append(f"noted: '{comments}'")
    
    # Combine into coherent summary
    if len(summary_parts) == 1:
        return summary_parts[0] + " provided feedback."
    elif len(summary_parts) == 2:
        return f"{summary_parts[0]} {summary_parts[1]}."
    else:
        return f"{summary_parts[0]} {summary_parts[1]}, {', '.join(summary_parts[2:-1])}, and {summary_parts[-1]}."

def get_satisfaction_level(rating):
    """
    Convert numerical satisfaction rating to descriptive level.
    
    Args:
        rating (float): Satisfaction rating
        
    Returns:
        str: Descriptive satisfaction level
    """
    
    if rating >= 4.5:
        return "very satisfied"
    elif rating >= 3.5:
        return "satisfied"
    elif rating >= 2.5:
        return "neutral"
    elif rating >= 1.5:
        return "dissatisfied"
    else:
        return "very dissatisfied"

def extract_ratings(row):
    """
    Extract all rating information from a survey response.
    
    Args:
        row (pd.Series): Survey response row
        
    Returns:
        dict: Ratings dictionary
    """
    
    ratings = {}
    
    # Extract all columns that contain 'rating' or 'satisfaction'
    for col in row.index:
        if 'rating' in col.lower() or 'satisfaction' in col.lower():
            if pd.notna(row[col]):
                ratings[col] = row[col]
    
    return ratings

def extract_improvement_areas(row):
    """
    Extract improvement areas from survey response.
    
    Args:
        row (pd.Series): Survey response row
        
    Returns:
        list: List of improvement areas
    """
    
    areas = []
    
    # Check for specific improvement area columns
    improvement_columns = [col for col in row.index if 'improvement' in col.lower() or 'area' in col.lower()]
    
    for col in improvement_columns:
        if pd.notna(row[col]) and str(row[col]).strip():
            areas.append(str(row[col]).strip())
    
    return areas

def extract_sentiment_indicators(row):
    """
    Extract sentiment indicators from survey response.
    
    Args:
        row (pd.Series): Survey response row
        
    Returns:
        dict: Sentiment indicators
    """
    
    indicators = {}
    
    # Analyze comments for sentiment indicators
    if 'comments' in row and pd.notna(row['comments']):
        comments = str(row['comments']).lower()
        
        # Positive indicators
        positive_words = ['good', 'great', 'excellent', 'love', 'happy', 'satisfied', 'pleased']
        positive_count = sum(1 for word in positive_words if word in comments)
        
        # Negative indicators
        negative_words = ['bad', 'terrible', 'hate', 'unhappy', 'dissatisfied', 'disappointed', 'poor']
        negative_count = sum(1 for word in negative_words if word in comments)
        
        indicators['positive_words'] = positive_count
        indicators['negative_words'] = negative_count
        indicators['sentiment_balance'] = positive_count - negative_count
    
    return indicators

def calculate_priority_score(row):
    """
    Calculate a priority score for follow-up based on survey response.
    
    Args:
        row (pd.Series): Survey response row
        
    Returns:
        float: Priority score (higher = more urgent)
    """
    
    score = 0.0
    
    # Low satisfaction increases priority
    if 'overall_satisfaction' in row and pd.notna(row['overall_satisfaction']):
        score += (5 - row['overall_satisfaction']) * 2
    
    # Negative sentiment in comments increases priority
    indicators = extract_sentiment_indicators(row)
    if 'sentiment_balance' in indicators:
        if indicators['sentiment_balance'] < 0:
            score += abs(indicators['sentiment_balance']) * 0.5
    
    # Recent surveys might be more urgent
    if 'survey_date' in row and pd.notna(row['survey_date']):
        try:
            survey_date = pd.to_datetime(row['survey_date'])
            days_ago = (datetime.now() - survey_date).days
            if days_ago < 7:
                score += 1.0  # Recent survey bonus
        except:
            pass
    
    return score

def generate_trend_analysis(df):
    """
    Generate trend analysis from survey data.
    
    Args:
        df (pd.DataFrame): Survey data
        
    Returns:
        dict: Trend analysis
    """
    
    trends = {}
    
    # Satisfaction trends over time
    if 'survey_date' in df.columns and 'overall_satisfaction' in df.columns:
        df['survey_date'] = pd.to_datetime(df['survey_date'])
        monthly_trends = df.groupby(df['survey_date'].dt.to_period('M'))['overall_satisfaction'].mean()
        trends['monthly_satisfaction'] = monthly_trends.to_dict()
    
    # Most common improvement areas
    if 'improvement_area' in df.columns:
        improvement_trends = df['improvement_area'].value_counts().head(10)
        trends['top_improvement_areas'] = improvement_trends.to_dict()
    
    # Rating correlations
    rating_columns = [col for col in df.columns if 'rating' in col.lower()]
    if len(rating_columns) > 1:
        correlation_matrix = df[rating_columns].corr()
        trends['rating_correlations'] = correlation_matrix.to_dict()
    
    return trends

def analyze_demographics(df):
    """
    Analyze demographic information from survey data.
    
    Args:
        df (pd.DataFrame): Survey data
        
    Returns:
        dict: Demographic analysis
    """
    
    demographics = {}
    
    # Look for common demographic columns
    demographic_columns = ['age_group', 'gender', 'location', 'customer_segment']
    
    for col in demographic_columns:
        if col in df.columns:
            demographics[col] = df[col].value_counts().to_dict()
    
    return demographics

def analyze_rating_breakdown(df):
    """
    Analyze rating breakdown across different categories.
    
    Args:
        df (pd.DataFrame): Survey data
        
    Returns:
        dict: Rating breakdown analysis
    """
    
    breakdown = {}
    
    # Find all rating columns
    rating_columns = [col for col in df.columns if 'rating' in col.lower()]
    
    for col in rating_columns:
        if col in df.columns:
            breakdown[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'distribution': df[col].value_counts().to_dict()
            }
    
    return breakdown

def analyze_temporal_trends(df):
    """
    Analyze temporal trends in survey data.
    
    Args:
        df (pd.DataFrame): Survey data
        
    Returns:
        dict: Temporal trends
    """
    
    trends = {}
    
    if 'survey_date' in df.columns:
        df['survey_date'] = pd.to_datetime(df['survey_date'])
        
        # Response volume over time
        daily_responses = df.groupby(df['survey_date'].dt.date).size()
        trends['daily_response_volume'] = daily_responses.to_dict()
        
        # Day of week analysis
        df['day_of_week'] = df['survey_date'].dt.day_name()
        day_trends = df['day_of_week'].value_counts()
        trends['day_of_week_distribution'] = day_trends.to_dict()
    
    return trends

def calculate_response_rate(df):
    """
    Calculate response rate metrics.
    
    Args:
        df (pd.DataFrame): Survey data
        
    Returns:
        dict: Response rate information
    """
    
    # This is a simplified calculation
    # In practice, you'd need to know how many surveys were sent vs received
    
    total_fields = len(df.columns)
    non_null_responses = df.count().sum()
    total_possible_responses = len(df) * total_fields
    
    completion_rate = non_null_responses / total_possible_responses if total_possible_responses > 0 else 0
    
    return {
        'completion_rate': completion_rate,
        'avg_fields_completed': completion_rate * total_fields
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()
    
    process_survey_data(args.input_path, args.output_path)