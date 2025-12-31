#!/usr/bin/env python3
"""
AWS Glue Data Quality Ruleset Creation Script

This script creates data quality rulesets for customer reviews data validation.
"""

import boto3
from awsglue.data_quality import DataQualityRule, DataQualityRulesetEvaluator

def create_customer_reviews_ruleset():
    """
    Create a data quality ruleset for customer reviews data.
    
    Returns:
        dict: Response from AWS Glue create_data_quality_ruleset API call
    """
    
    # Define rules for customer reviews
    rules = [
        # Check for completeness of required fields
        DataQualityRule.is_complete("review_text"),
        DataQualityRule.is_complete("product_id"),
        DataQualityRule.is_complete("customer_id"),
        
        # Check for valid values
        DataQualityRule.column_values_match_pattern("review_text", ".{10,}"),  # At least 10 chars
        DataQualityRule.column_values_match_pattern("rating", "^[1-5]$"),  # Rating 1-5
        
        # Check for data consistency
        DataQualityRule.column_values_match_pattern("review_date", "\\d{4}-\\d{2}-\\d{2}"),  # YYYY-MM-DD
        
        # Check for statistical properties
        DataQualityRule.column_length_distribution_match("review_text", 
                                                        min_length=10, 
                                                        max_length=5000)
    ]
    
    # Create ruleset
    glue_client = boto3.client('glue')
    response = glue_client.create_data_quality_ruleset(
        Name='customer_reviews_ruleset',
        Description='Data quality rules for customer reviews',
        Ruleset='\n'.join([str(rule) for rule in rules]),
        Tags={'Project': 'CustomerFeedbackAnalysis'}
    )
    
    print(f"Created ruleset: {response['Name']}")
    return response

def create_survey_data_ruleset():
    """
    Create a data quality ruleset for survey data.
    
    Returns:
        dict: Response from AWS Glue create_data_quality_ruleset API call
    """
    
    rules = [
        # Check for completeness of required fields
        DataQualityRule.is_complete("customer_id"),
        DataQualityRule.is_complete("survey_date"),
        DataQualityRule.is_complete("overall_satisfaction"),
        
        # Check for valid values
        DataQualityRule.column_values_in_set("overall_satisfaction", 
                                            ["Very Dissatisfied", "Dissatisfied", 
                                             "Neutral", "Satisfied", "Very Satisfied"]),
        
        # Check for data consistency
        DataQualityRule.column_values_match_pattern("survey_date", "\\d{4}-\\d{2}-\\d{2}"),  # YYYY-MM-DD
        DataQualityRule.is_unique("customer_id"),  # Unique customer IDs
        
        # Check for statistical properties
        DataQualityRule.column_length_distribution_match("improvement_area", 
                                                        min_length=5, 
                                                        max_length=100)
    ]
    
    # Create ruleset
    glue_client = boto3.client('glue')
    response = glue_client.create_data_quality_ruleset(
        Name='survey_data_ruleset',
        Description='Data quality rules for survey data',
        Ruleset='\n'.join([str(rule) for rule in rules]),
        Tags={'Project': 'CustomerFeedbackAnalysis'}
    )
    
    print(f"Created ruleset: {response['Name']}")
    return response

def evaluate_data_quality(database_name, table_name, ruleset_name):
    """
    Evaluate data quality for a specific table using a ruleset.
    
    Args:
        database_name (str): Name of the Glue database
        table_name (str): Name of the table to evaluate
        ruleset_name (str): Name of the ruleset to use
        
    Returns:
        dict: Data quality evaluation results
    """
    
    evaluator = DataQualityRulesetEvaluator()
    
    # Run data quality evaluation
    results = evaluator.evaluate_ruleset(
        database_name=database_name,
        table_name=table_name,
        ruleset_name=ruleset_name
    )
    
    print(f"Data quality evaluation for {table_name}:")
    print(f"Overall score: {results.get('overall_score', 'N/A')}")
    print(f"Rules passed: {results.get('rules_passed', 0)}")
    print(f"Rules failed: {results.get('rules_failed', 0)}")
    
    return results

if __name__ == "__main__":
    # Create the rulesets
    try:
        print("Creating customer reviews ruleset...")
        create_customer_reviews_ruleset()
        
        print("\nCreating survey data ruleset...")
        create_survey_data_ruleset()
        
        print("\nRulesets created successfully!")
        
    except Exception as e:
        print(f"Error creating rulesets: {str(e)}")