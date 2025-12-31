#!/usr/bin/env python3
"""
AWS Glue Data Quality Ruleset Creation Script for LM Project

This script creates data quality rulesets for customer reviews data validation
specifically for the lm-ai-feedback-dev bucket.
"""

import boto3
import json

def create_customer_reviews_ruleset():
    """
    Create a data quality ruleset for customer reviews data.
    
    Returns:
        dict: Response from AWS Glue create_data_quality_ruleset API call
    """
    
    # Define rules for customer reviews
    ruleset_json = {
        "Rules": [
            {
                "Name": "review_text_not_null",
                "Description": "Review text should not be null",
                "RuleType": "IsNotNull",
                "ColumnName": "review_text"
            },
            {
                "Name": "customer_id_not_null",
                "Description": "Customer ID should not be null",
                "RuleType": "IsNotNull",
                "ColumnName": "customer_id"
            },
            {
                "Name": "rating_in_range",
                "Description": "Rating should be between 1 and 5",
                "RuleType": "Expression",
                "Expression": "rating >= 1 AND rating <= 5"
            },
            {
                "Name": "review_text_length",
                "Description": "Review text should be at least 10 characters",
                "RuleType": "Expression",
                "Expression": "length(review_text) >= 10"
            },
            {
                "Name": "review_date_format",
                "Description": "Review date should be in valid date format",
                "RuleType": "RegexMatch",
                "ColumnName": "review_date",
                "Pattern": "\\d{4}-\\d{2}-\\d{2}"
            }
        ]
    }
    
    # Create ruleset
    glue_client = boto3.client('glue')
    response = glue_client.create_data_quality_ruleset(
        Name='customer_reviews_lm_ruleset',
        Description='Data quality rules for customer reviews - LM Project',
        Ruleset=json.dumps(ruleset_json),
        Tags={'Project': 'CustomerFeedbackLM'}
    )
    
    print(f"Created ruleset: {response['Name']}")
    return response

def create_survey_data_ruleset():
    """
    Create a data quality ruleset for survey data.
    
    Returns:
        dict: Response from AWS Glue create_data_quality_ruleset API call
    """
    
    ruleset_json = {
        "Rules": [
            {
                "Name": "customer_id_not_null",
                "Description": "Customer ID should not be null",
                "RuleType": "IsNotNull",
                "ColumnName": "customer_id"
            },
            {
                "Name": "survey_date_not_null",
                "Description": "Survey date should not be null",
                "RuleType": "IsNotNull",
                "ColumnName": "survey_date"
            },
            {
                "Name": "satisfaction_in_set",
                "Description": "Overall satisfaction should be in valid set",
                "RuleType": "AllowedValues",
                "ColumnName": "overall_satisfaction",
                "AllowedValues": ["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"]
            },
            {
                "Name": "recommendation_score_range",
                "Description": "Recommendation score should be between 0 and 10",
                "RuleType": "Expression",
                "Expression": "recommendation_score >= 0 AND recommendation_score <= 10"
            },
            {
                "Name": "survey_date_format",
                "Description": "Survey date should be in valid date format",
                "RuleType": "RegexMatch",
                "ColumnName": "survey_date",
                "Pattern": "\\d{4}-\\d{2}-\\d{2}"
            }
        ]
    }
    
    # Create ruleset
    glue_client = boto3.client('glue')
    response = glue_client.create_data_quality_ruleset(
        Name='survey_data_lm_ruleset',
        Description='Data quality rules for survey data - LM Project',
        Ruleset=json.dumps(ruleset_json),
        Tags={'Project': 'CustomerFeedbackLM'}
    )
    
    print(f"Created ruleset: {response['Name']}")
    return response

def create_image_metadata_ruleset():
    """
    Create a data quality ruleset for image metadata.
    
    Returns:
        dict: Response from AWS Glue create_data_quality_ruleset API call
    """
    
    ruleset_json = {
        "Rules": [
            {
                "Name": "customer_id_not_null",
                "Description": "Customer ID should not be null",
                "RuleType": "IsNotNull",
                "ColumnName": "customer_id"
            },
            {
                "Name": "image_path_not_null",
                "Description": "Image path should not be null",
                "RuleType": "IsNotNull",
                "ColumnName": "image_path"
            },
            {
                "Name": "file_format_valid",
                "Description": "File format should be valid image format",
                "RuleType": "AllowedValues",
                "ColumnName": "file_format",
                "AllowedValues": ["jpg", "jpeg", "png", "webp"]
            },
            {
                "Name": "file_size_positive",
                "Description": "File size should be positive",
                "RuleType": "Expression",
                "Expression": "file_size > 0"
            }
        ]
    }
    
    # Create ruleset
    glue_client = boto3.client('glue')
    response = glue_client.create_data_quality_ruleset(
        Name='image_metadata_lm_ruleset',
        Description='Data quality rules for image metadata - LM Project',
        Ruleset=json.dumps(ruleset_json),
        Tags={'Project': 'CustomerFeedbackLM'}
    )
    
    print(f"Created ruleset: {response['Name']}")
    return response

def create_audio_metadata_ruleset():
    """
    Create a data quality ruleset for audio metadata.
    
    Returns:
        dict: Response from AWS Glue create_data_quality_ruleset API call
    """
    
    ruleset_json = {
        "Rules": [
            {
                "Name": "customer_id_not_null",
                "Description": "Customer ID should not be null",
                "RuleType": "IsNotNull",
                "ColumnName": "customer_id"
            },
            {
                "Name": "audio_path_not_null",
                "Description": "Audio path should not be null",
                "RuleType": "IsNotNull",
                "ColumnName": "audio_path"
            },
            {
                "Name": "file_format_valid",
                "Description": "File format should be valid audio format",
                "RuleType": "AllowedValues",
                "ColumnName": "file_format",
                "AllowedValues": ["mp3", "wav", "flac", "m4a"]
            },
            {
                "Name": "duration_positive",
                "Description": "Duration should be positive",
                "RuleType": "Expression",
                "Expression": "duration_seconds > 0"
            }
        ]
    }
    
    # Create ruleset
    glue_client = boto3.client('glue')
    response = glue_client.create_data_quality_ruleset(
        Name='audio_metadata_lm_ruleset',
        Description='Data quality rules for audio metadata - LM Project',
        Ruleset=json.dumps(ruleset_json),
        Tags={'Project': 'CustomerFeedbackLM'}
    )
    
    print(f"Created ruleset: {response['Name']}")
    return response

if __name__ == "__main__":
    # Create the rulesets
    try:
        print("Creating customer reviews ruleset...")
        create_customer_reviews_ruleset()
        
        print("\nCreating survey data ruleset...")
        create_survey_data_ruleset()
        
        print("\nCreating image metadata ruleset...")
        create_image_metadata_ruleset()
        
        print("\nCreating audio metadata ruleset...")
        create_audio_metadata_ruleset()
        
        print("\nAll rulesets created successfully!")
        
    except Exception as e:
        print(f"Error creating rulesets: {str(e)}")